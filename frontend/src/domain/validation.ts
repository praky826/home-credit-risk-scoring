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

function validateScore(
  errors: ValidationErrors,
  field: keyof ApplicantFormValues,
  value: string,
  label: string,
) {
  const parsed = parseOptionalNumber(value);
  if (parsed === undefined) {
    return;
  }

  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    errors[field] = `${label} must be between 0 and 1.`;
  }
}

export function validateApplicantForm(values: ApplicantFormValues): ValidationErrors {
  const errors: ValidationErrors = {};

  setPositiveError(errors, "ageYears", values.ageYears, "Age", true);
  setPositiveError(errors, "annualIncome", values.annualIncome, "Annual income", true);
  setPositiveError(errors, "loanAmount", values.loanAmount, "Loan amount", true);

  const age = parseOptionalNumber(values.ageYears);
  const yearsEmployed = parseOptionalNumber(values.yearsEmployed);

  if (age !== undefined && (age < 18 || age > 100)) {
    errors.ageYears = "Age must be between 18 and 100.";
  }

  if (yearsEmployed === undefined) {
    errors.yearsEmployed = "Years employed is required.";
  } else if (!Number.isFinite(yearsEmployed) || yearsEmployed < 0) {
    errors.yearsEmployed = "Years employed cannot be negative.";
  } else if (age !== undefined && yearsEmployed > age - 14) {
    errors.yearsEmployed = "Years employed cannot be greater than working age.";
  }

  validateScore(errors, "externalScore1", values.externalScore1, "Credit bureau score 1");
  validateScore(errors, "externalScore2", values.externalScore2, "Credit bureau score 2");
  validateScore(errors, "externalScore3", values.externalScore3, "Credit bureau score 3");

  setPositiveError(errors, "annuityAmount", values.annuityAmount, "Annuity amount");
  setPositiveError(errors, "goodsPrice", values.goodsPrice, "Goods price");

  const childrenCount = parseOptionalNumber(values.childrenCount);
  if (childrenCount !== undefined && (!Number.isInteger(childrenCount) || childrenCount < 0)) {
    errors.childrenCount = "Number of children must be a whole number 0 or greater.";
  }

  const familyMembers = parseOptionalNumber(values.familyMembers);
  if (familyMembers !== undefined && familyMembers < 0) {
    errors.familyMembers = "Family members cannot be negative.";
  }

  return errors;
}

export function hasValidationErrors(errors: ValidationErrors): boolean {
  return Object.keys(errors).length > 0;
}
