export function normalizeLoanId(value: string): string {
  const normalized = value.trim();
  if (!normalized) {
    throw new Error("Loan ID is required.");
  }
  if (!/^[A-Za-z0-9_-]+$/.test(normalized)) {
    throw new Error("Loan ID may contain only letters, numbers, hyphens, and underscores.");
  }
  return normalized;
}
