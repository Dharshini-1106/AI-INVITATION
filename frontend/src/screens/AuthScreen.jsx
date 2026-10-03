import React, { useState } from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { login, signup } from '../services/api';
import { useAuth } from '../auth/AuthContext';

export default function AuthScreen({ mode }) {
  const registering = mode === 'signup';
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm_password: '' });
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { user, setUser, loading } = useAuth();
  if (!loading && user && !registering) return <Navigate to="/home" replace />;
  const change = (event) => setForm({ ...form, [event.target.name]: event.target.value });
  const submit = async (event) => {
    event.preventDefault(); setError('');
    if (registering && form.password !== form.confirm_password) { setError('Passwords do not match.'); return; }
    if (registering && (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{12,}/.test(form.password))) { setError('Use at least 12 characters with uppercase, lowercase, and a number.'); return; }
    setPending(true);
    try {
      if (registering) { await signup(form); navigate('/login', { replace: true, state: { notice: 'Account created. Please log in.' } }); }
      else { setUser(await login(form.email, form.password)); navigate(location.state?.from || '/home', { replace: true }); }
    } catch (e) { setError(e.response?.data?.detail || 'Unable to complete your request. Please try again.'); }
    finally { setPending(false); }
  };
  return <main className="auth-page"><form className="auth-card" onSubmit={submit}>
    <div className="auth-mark">✉</div><p className="auth-eyebrow">INVITATIONSENSE</p>
    <h1>{registering ? 'Create your account' : 'Welcome back'}</h1>
    <p className="auth-subtitle">{registering ? 'Sign up to start understanding your invitations.' : 'Log in to continue to your invitations.'}</p>
    {location.state?.notice && !registering && <div className="auth-success">{location.state.notice}</div>}
    {registering && <label>Full name<input name="name" autoComplete="name" value={form.name} onChange={change} required minLength="2" maxLength="100" /></label>}
    <label>Email address<input name="email" type="email" autoComplete="email" value={form.email} onChange={change} required /></label>
    <label>Password<input name="password" type="password" autoComplete={registering ? 'new-password' : 'current-password'} value={form.password} onChange={change} required minLength={registering ? 12 : 1} maxLength="128" /></label>
    {registering && <label>Confirm password<input name="confirm_password" type="password" autoComplete="new-password" value={form.confirm_password} onChange={change} required maxLength="128" /></label>}
    {error && <div className="auth-error" role="alert">{error}</div>}
    <button className="auth-submit" type="submit" disabled={pending}>{pending ? 'Please wait…' : registering ? 'Sign Up' : 'Log In'}</button>
    <p className="auth-switch">{registering ? 'Already have an account?' : 'New to InvitationSense?'}{' '}
      <Link to={registering ? '/login' : '/signup'}>{registering ? 'Log in' : 'Sign up'}</Link></p>
  </form></main>;
}
