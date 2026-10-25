import type { ApplicantFormValues } from "./types";

export type ValidationErrors = Partial<Record<keyof ApplicantFormValues, string>>;

function parseOptionalNumber(value: string): number | undefined {
  if (value.trim() === "") {
    return undefined;
  }

  return Number(value);
}

function setPositiveError(
  errors: ValidationErrors,
  field: keyof ApplicantFormValues,
  value: string,
  label: string,
  required = false,
) {
  const parsed = parseOptionalNumber(value);

  if (parsed === undefined) {
    if (required) {
      errors[field] = `${label} is required.`;
    }
    return;
  }

  if (!Number.isFinite(parsed) || parsed <= 0) {
    errors[field] = `${label} must be greater than 0.`;
  }
}

function validateFico(
  errors: ValidationErrors,
  field: keyof ApplicantFormValues,
  value: string,
  label: string,
) {
  const parsed = parseOptionalNumber(value);
  if (parsed === undefined) {
    // Optional in Alternative / Cashflow-First mode
    return;
  }

  if (!Number.isFinite(parsed) || parsed < 300 || parsed > 850) {
    errors[field] = `${label} must be between 300 and 850.`;
  }
}

export function validateApplicantForm(values: ApplicantFormValues): ValidationErrors {
  const errors: ValidationErrors = {};

  setPositiveError(errors, "annualIncome", values.annualIncome, "Annual income", true);
  setPositiveError(errors, "loanAmount", values.loanAmount, "Loan amount", true);
  setPositiveError(errors, "term", values.term, "Loan term", true);
  setPositiveError(errors, "installment", values.installment, "Monthly installment", false);
  setPositiveError(errors, "dti", values.dti, "Debt-to-income ratio", true);

  validateFico(errors, "ficoRangeLow", values.ficoRangeLow, "FICO low");
  validateFico(errors, "ficoRangeHigh", values.ficoRangeHigh, "FICO high");

  const loanAmount = parseOptionalNumber(values.loanAmount);
  if (loanAmount !== undefined) {
    if (loanAmount < 500) {
      errors.loanAmount = "Minimum unsecured loan amount is $500.";
    } else if (loanAmount > 40000) {
      errors.loanAmount = "Loan amount cannot exceed $40,000 (LendingClub maximum limit).";
    }
  }

  const dti = parseOptionalNumber(values.dti);
  if (dti !== undefined && (dti < 0 || dti > 100)) {
    errors.dti = "Debt-to-income ratio must be between 0% and 100%.";
  }

  const revolvingUtilization = parseOptionalNumber(values.revolvingUtilization);
  if (revolvingUtilization !== undefined && (revolvingUtilization < 0 || revolvingUtilization > 100)) {
    errors.revolvingUtilization = "Revolving utilization must be between 0% and 100%.";
  }

  const term = parseOptionalNumber(values.term);
  if (term !== undefined && term !== 36 && term !== 60) {
    errors.term = "Loan term must be 36 or 60 months.";
  }

  const ficoLow = parseOptionalNumber(values.ficoRangeLow);
  const ficoHigh = parseOptionalNumber(values.ficoRangeHigh);
  if (ficoLow !== undefined && ficoHigh !== undefined && ficoLow > ficoHigh) {
    errors.ficoRangeLow = "FICO low cannot be greater than FICO high.";
  }

  const empLength = parseOptionalNumber(values.empLength);
  if (empLength !== undefined && (!Number.isFinite(empLength) || empLength < 0 || empLength > 10)) {
    errors.empLength = "Employment length must be between 0 and 10 years.";
  }

  const integerFields: Array<[keyof ApplicantFormValues, string]> = [
    ["delinq2Yrs", "Delinquencies"],
    ["inquiriesLast6Months", "Inquiries"],
    ["publicRecords", "Public records"],
    ["openAccounts", "Open accounts"],
    ["totalAccounts", "Total accounts"],
    ["mortgageAccounts", "Mortgage accounts"],
    ["publicRecordBankruptcies", "Public bankruptcies"],
  ];
  for (const [field, label] of integerFields) {
    const value = parseOptionalNumber(values[field]);
    if (value !== undefined && (!Number.isInteger(value) || value < 0)) {
      errors[field] = `${label} must be a whole number 0 or greater.`;
    }
  }

  return errors;
}

export function hasValidationErrors(errors: ValidationErrors): boolean {
  return Object.keys(errors).length > 0;
}
