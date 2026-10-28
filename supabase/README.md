# Supabase persistence setup

The migration in `migrations/20261028000000_underwriting_persistence.sql` adds owner-scoped application drafts, current underwriting decisions, and trigger-generated decision history.

Apply it in the Supabase SQL Editor before testing Save draft or Save decision. The frontend stores loan IDs as text, so values such as `001` retain their leading zeroes.

The frontend uses only the Supabase publishable key. Never put a service-role key in `frontend/.env.local` or any Vite variable.
