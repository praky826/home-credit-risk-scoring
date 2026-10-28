import { useState } from "react";
import { Eye, EyeOff, Info, LockKeyhole, Mail, ShieldCheck } from "lucide-react";
import { useAuth } from "../../auth/AuthProvider";

export function LoginPage() {
  const { signIn, requestPasswordReset, error: authError } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isResetMode, setIsResetMode] = useState(false);
  const [message, setMessage] = useState("");
  const [fieldError, setFieldError] = useState("");

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFieldError("");
    setMessage("");

    if (!email.trim() || (!isResetMode && !password)) {
      setFieldError(isResetMode ? "Enter your email address." : "Enter your email address and password.");
      return;
    }

    setIsSubmitting(true);
    const succeeded = isResetMode
      ? await requestPasswordReset(email.trim())
      : await signIn(email.trim(), password);
    setIsSubmitting(false);

    if (succeeded && isResetMode) {
      setMessage("If an administrator-managed account exists for this email, a reset link has been sent.");
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-card" aria-labelledby="login-title">
        <div className="auth-brand">
          <span className="auth-brand-mark"><ShieldCheck size={18} /></span>
          <span>LendingClub <strong>Risk Intelligence</strong></span>
        </div>
        <div className="auth-heading">
          <p className="eyebrow">Protected underwriting workspace</p>
          <h1 id="login-title">{isResetMode ? "Reset your password" : "Welcome back"}</h1>
          <p>{isResetMode ? "Enter your email to receive a secure reset link." : "Sign in to continue to the risk intelligence console"}</p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit} noValidate>
          <label className="auth-field">
            <span>Email address</span>
            <span className="auth-input-shell">
              <Mail size={17} aria-hidden="true" />
              <input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Enter your email address" aria-invalid={Boolean(fieldError)} />
            </span>
          </label>

          {!isResetMode && <label className="auth-field">
            <span>Password</span>
            <span className="auth-input-shell">
              <LockKeyhole size={17} aria-hidden="true" />
              <input type={showPassword ? "text" : "password"} autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Enter your password" aria-invalid={Boolean(fieldError)} />
              <button type="button" className="password-toggle" onClick={() => setShowPassword((current) => !current)} aria-label={showPassword ? "Hide password" : "Show password"}>
                {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
              </button>
            </span>
          </label>}

          {(fieldError || authError) && <p className="auth-error" role="alert">{fieldError || authError}</p>}
          {message && <p className="auth-success" role="status">{message}</p>}

          <button className="auth-submit" type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Working..." : isResetMode ? "Send reset link" : "Sign in"}
          </button>
        </form>

        <button type="button" className="auth-link" onClick={() => { setIsResetMode((current) => !current); setMessage(""); setFieldError(""); }}>
          {isResetMode ? "Back to sign in" : "Forgot password?"}
        </button>

        <div className="auth-notice">
          <Info size={16} aria-hidden="true" />
          <p>Access is provided by your administrator.<br />New account registration is disabled.</p>
        </div>
      </section>
      <p className="auth-footer">Protected underwriting workspace · Administrator-managed access</p>
    </main>
  );
}
