create table if not exists public.application_drafts (
  id uuid primary key default gen_random_uuid(),
  loan_id text not null,
  officer_user_id uuid not null references auth.users(id) on delete cascade,
  form_values jsonb not null,
  workflow_stage text not null default 'draft',
  risk_result jsonb,
  risk_is_stale boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint application_drafts_loan_id_trimmed check (loan_id = btrim(loan_id)),
  constraint application_drafts_workflow_stage_valid check (workflow_stage in ('draft', 'details_complete', 'risk_assessed', 'decision_recorded')),
  unique (officer_user_id, loan_id)
);

create index if not exists application_drafts_officer_updated_idx
  on public.application_drafts (officer_user_id, updated_at desc);

create table if not exists public.underwriting_decisions (
  id uuid primary key default gen_random_uuid(),
  loan_id text not null,
  officer_user_id uuid not null references auth.users(id) on delete cascade,
  system_recommendation text not null,
  officer_decision text not null,
  override_reason text,
  prediction_request_id text,
  model_version text,
  default_probability numeric,
  decision_version integer not null default 1,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint underwriting_decisions_loan_id_trimmed check (loan_id = btrim(loan_id)),
  constraint underwriting_decisions_system_recommendation_valid check (system_recommendation in ('approve', 'refer', 'decline')),
  constraint underwriting_decisions_officer_decision_valid check (officer_decision in ('approve', 'refer', 'decline')),
  constraint underwriting_decisions_override_required check (
    officer_decision = system_recommendation
    or nullif(btrim(override_reason), '') is not null
  ),
  unique (officer_user_id, loan_id)
);

create index if not exists underwriting_decisions_officer_loan_idx
  on public.underwriting_decisions (officer_user_id, loan_id);

create table if not exists public.decision_audit_events (
  id uuid primary key default gen_random_uuid(),
  decision_id uuid not null references public.underwriting_decisions(id) on delete cascade,
  loan_id text not null,
  officer_user_id uuid not null references auth.users(id) on delete cascade,
  event_type text not null,
  previous_snapshot jsonb,
  new_snapshot jsonb not null,
  created_at timestamptz not null default now(),
  constraint decision_audit_events_type_valid check (event_type in ('created', 'updated'))
);

create index if not exists decision_audit_events_officer_loan_idx
  on public.decision_audit_events (officer_user_id, loan_id, created_at desc);

create or replace function public.set_underwriting_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create or replace function public.prepare_decision_update()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at = now();
  new.decision_version = old.decision_version + 1;
  return new;
end;
$$;

create or replace function public.record_decision_audit_event()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.decision_audit_events (
    decision_id,
    loan_id,
    officer_user_id,
    event_type,
    previous_snapshot,
    new_snapshot
  ) values (
    new.id,
    new.loan_id,
    new.officer_user_id,
    case when tg_op = 'INSERT' then 'created' else 'updated' end,
    case when tg_op = 'INSERT' then null else jsonb_build_object(
      'system_recommendation', old.system_recommendation,
      'officer_decision', old.officer_decision,
      'override_reason', old.override_reason,
      'prediction_request_id', old.prediction_request_id,
      'model_version', old.model_version,
      'default_probability', old.default_probability,
      'decision_version', old.decision_version,
      'updated_at', old.updated_at
    ) end,
    jsonb_build_object(
      'system_recommendation', new.system_recommendation,
      'officer_decision', new.officer_decision,
      'override_reason', new.override_reason,
      'prediction_request_id', new.prediction_request_id,
      'model_version', new.model_version,
      'default_probability', new.default_probability,
      'decision_version', new.decision_version,
      'updated_at', new.updated_at
    )
  );
  return new;
end;
$$;

create or replace trigger application_drafts_updated_at
before update on public.application_drafts
for each row execute function public.set_underwriting_updated_at();

create or replace trigger underwriting_decisions_prepare_update
before update on public.underwriting_decisions
for each row execute function public.prepare_decision_update();

create or replace trigger underwriting_decisions_audit
after insert or update on public.underwriting_decisions
for each row execute function public.record_decision_audit_event();

alter table public.application_drafts enable row level security;
alter table public.underwriting_decisions enable row level security;
alter table public.decision_audit_events enable row level security;

drop policy if exists "officers read their own drafts" on public.application_drafts;
drop policy if exists "officers create their own drafts" on public.application_drafts;
drop policy if exists "officers update their own drafts" on public.application_drafts;
create policy "officers read their own drafts"
on public.application_drafts for select to authenticated
using (officer_user_id = auth.uid());
create policy "officers create their own drafts"
on public.application_drafts for insert to authenticated
with check (officer_user_id = auth.uid());
create policy "officers update their own drafts"
on public.application_drafts for update to authenticated
using (officer_user_id = auth.uid())
with check (officer_user_id = auth.uid());

drop policy if exists "officers read their own decisions" on public.underwriting_decisions;
drop policy if exists "officers create their own decisions" on public.underwriting_decisions;
drop policy if exists "officers update their own decisions" on public.underwriting_decisions;
create policy "officers read their own decisions"
on public.underwriting_decisions for select to authenticated
using (officer_user_id = auth.uid());
create policy "officers create their own decisions"
on public.underwriting_decisions for insert to authenticated
with check (officer_user_id = auth.uid());
create policy "officers update their own decisions"
on public.underwriting_decisions for update to authenticated
using (officer_user_id = auth.uid())
with check (officer_user_id = auth.uid());

drop policy if exists "officers read their own decision history" on public.decision_audit_events;
create policy "officers read their own decision history"
on public.decision_audit_events
for select to authenticated
using (officer_user_id = auth.uid());
