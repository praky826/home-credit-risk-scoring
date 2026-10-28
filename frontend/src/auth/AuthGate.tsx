import { Loader2 } from "lucide-react";
import App from "../App";
import { LoginPage } from "../components/auth/LoginPage";
import { useAuth } from "./AuthProvider";

export function AuthGate() {
  const { session, isLoading } = useAuth();

  if (isLoading) {
    return <main className="auth-loading" aria-live="polite"><Loader2 className="spin" size={22} /> Restoring secure session...</main>;
  }

  return session ? <App /> : <LoginPage />;
}
