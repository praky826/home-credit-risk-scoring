import type { ApplicantFormValues, PredictPayload } from "./types";

function optionalNumber(value: string): number | undefined {
  if (value.trim() === "") {
    return undefined;
  }

  return Number(value);
}

function optionalInteger(value: string): number | undefined {
  const parsed = optionalNumber(value);
  return parsed === undefined ? undefined : Math.round(parsed);
}

export function mapFormToPredictPayload(values: ApplicantFormValues): PredictPayload {
  const payload: PredictPayload = {
    loan_amnt: Number(values.loanAmount),
    annual_inc: Number(values.annualIncome),
    term: Number(values.term),
    installment: Number(values.installment),
    dti: Number(values.dti),
  };

  const ficoLow = optionalNumber(values.ficoRangeLow);
  if (ficoLow !== undefined) {
    payload.fico_range_low = ficoLow;
  }
  const ficoHigh = optionalNumber(values.ficoRangeHigh);
  if (ficoHigh !== undefined) {
    payload.fico_range_high = ficoHigh;
  }

  const optionalFields: Partial<PredictPayload> = {
    loan_id: optionalInteger(values.loanId),
    emp_length: optionalNumber(values.empLength),
    home_ownership: values.homeOwnership || undefined,
    verification_status: values.verificationStatus || undefined,
    purpose: values.purpose || undefined,
    delinq_2yrs: optionalInteger(values.delinq2Yrs),
    inq_last_6mths: optionalInteger(values.inquiriesLast6Months),
    pub_rec: optionalInteger(values.publicRecords),
    revol_bal: optionalNumber(values.revolvingBalance),
    revol_util: optionalNumber(values.revolvingUtilization),
    open_acc: optionalInteger(values.openAccounts),
    total_acc: optionalInteger(values.totalAccounts),
    mort_acc: optionalInteger(values.mortgageAccounts),
    pub_rec_bankruptcies: optionalInteger(values.publicRecordBankruptcies),
  };

  for (const [key, value] of Object.entries(optionalFields)) {
    if (typeof value === "string" && value.trim() !== "") {
      payload[key as keyof PredictPayload] = value as never;
    } else if (typeof value === "number" && Number.isFinite(value)) {
      payload[key as keyof PredictPayload] = value as never;
    }
  }

  return payload;
}
