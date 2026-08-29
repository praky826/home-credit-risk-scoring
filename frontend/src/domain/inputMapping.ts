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

function yesNoToBinary(value: "" | "Y" | "N"): number | undefined {
  if (value === "") {
    return undefined;
  }

  return value === "Y" ? 1 : 0;
}

function genderToCode(value: "" | "F" | "M"): number | undefined {
  if (value === "") {
    return undefined;
  }

  return value === "M" ? 1 : 0;
}

export function mapFormToPredictPayload(values: ApplicantFormValues): PredictPayload {
  const payload: PredictPayload = {
    AMT_CREDIT: Number(values.loanAmount),
    AMT_INCOME_TOTAL: Number(values.annualIncome),
    DAYS_BIRTH: -Math.round(Number(values.ageYears) * 365),
    DAYS_EMPLOYED: -Math.round(Number(values.yearsEmployed) * 365),
  };

  const optionalFields: Partial<PredictPayload> = {
    SK_ID_CURR: optionalInteger(values.customerId),
    EXT_SOURCE_2: optionalNumber(values.externalScore2),
    AMT_ANNUITY: optionalNumber(values.annuityAmount),
    AMT_GOODS_PRICE: optionalNumber(values.goodsPrice),
    EXT_SOURCE_1: optionalNumber(values.externalScore1),
    EXT_SOURCE_3: optionalNumber(values.externalScore3),
    CODE_GENDER: genderToCode(values.gender),
    FLAG_OWN_CAR: yesNoToBinary(values.ownsCar),
    FLAG_OWN_REALTY: yesNoToBinary(values.ownsProperty),
    CNT_CHILDREN: optionalInteger(values.childrenCount),
    CNT_FAM_MEMBERS: optionalNumber(values.familyMembers),
  };

  for (const [key, value] of Object.entries(optionalFields)) {
    if (value !== undefined && Number.isFinite(value)) {
      payload[key as keyof PredictPayload] = value as never;
    }
  }

  return payload;
}
