import type { FeatureCatalogItem } from "./types";

const optionalFeatures: FeatureCatalogItem[] = [
  {
    key: "purpose",
    label: "Loan purpose",
    importance: 90,
    kind: "select",
    options: [
      { value: "", label: "Not provided" },
      { value: "debt_consolidation", label: "Debt consolidation" },
      { value: "credit_card", label: "Credit card refinance" },
      { value: "home_improvement", label: "Home improvement" },
      { value: "major_purchase", label: "Major purchase" },
      { value: "small_business", label: "Small business" },
      { value: "other", label: "Other" },
    ],
  },
  {
    key: "homeOwnership",
    label: "Home ownership",
    importance: 75,
    kind: "select",
    options: [
      { value: "", label: "Not provided" },
      { value: "RENT", label: "Rent" },
      { value: "OWN", label: "Own" },
      { value: "MORTGAGE", label: "Mortgage" },
      { value: "OTHER", label: "Other" },
    ],
  },
  {
    key: "verificationStatus",
    label: "Income verification",
    importance: 65,
    kind: "select",
    options: [
      { value: "", label: "Not provided" },
      { value: "Not Verified", label: "Not verified" },
      { value: "Source Verified", label: "Source verified" },
      { value: "Verified", label: "Verified" },
    ],
  },
  {
    key: "revolvingUtilization",
    label: "Revolving utilization",
    importance: 60,
    kind: "number",
  },
  {
    key: "revolvingBalance",
    label: "Revolving balance",
    importance: 45,
    kind: "number",
  },
  {
    key: "openAccounts",
    label: "Open credit accounts",
    importance: 35,
    kind: "number",
  },
  { key: "totalAccounts", label: "Total credit accounts", importance: 30, kind: "number" },
  { key: "delinq2Yrs", label: "Delinquencies in 2 years", importance: 28, kind: "number" },
  { key: "inquiriesLast6Months", label: "Inquiries in 6 months", importance: 24, kind: "number" },
  { key: "publicRecords", label: "Public records", importance: 20, kind: "number" },
  { key: "mortgageAccounts", label: "Mortgage accounts", importance: 18, kind: "number" },
  { key: "publicRecordBankruptcies", label: "Public bankruptcies", importance: 15, kind: "number" },
];

export const optionalFeatureCatalog = [...optionalFeatures].sort(
  (left, right) => right.importance - left.importance,
);
