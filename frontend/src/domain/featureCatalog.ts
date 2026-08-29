import type { FeatureCatalogItem } from "./types";

const optionalFeatures: FeatureCatalogItem[] = [
  {
    key: "goodsPrice",
    label: "Goods price",
    importance: 90,
    kind: "number",
  },
  {
    key: "gender",
    label: "Gender",
    importance: 75,
    kind: "select",
    options: [
      { value: "", label: "Not provided" },
      { value: "F", label: "Female" },
      { value: "M", label: "Male" },
    ],
  },
  {
    key: "ownsProperty",
    label: "Owns property",
    importance: 65,
    kind: "select",
    options: [
      { value: "", label: "Not provided" },
      { value: "Y", label: "Yes" },
      { value: "N", label: "No" },
    ],
  },
  {
    key: "ownsCar",
    label: "Owns car",
    importance: 60,
    kind: "select",
    options: [
      { value: "", label: "Not provided" },
      { value: "Y", label: "Yes" },
      { value: "N", label: "No" },
    ],
  },
  {
    key: "familyMembers",
    label: "Family members",
    importance: 45,
    kind: "number",
  },
  {
    key: "childrenCount",
    label: "Number of children",
    importance: 35,
    kind: "number",
  },
];

export const optionalFeatureCatalog = [...optionalFeatures].sort(
  (left, right) => right.importance - left.importance,
);
