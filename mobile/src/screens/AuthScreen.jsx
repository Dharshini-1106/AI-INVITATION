import React, { useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import colors from '../theme/colors';
import { login, signup } from '../services/api';

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: colors.background },
  content: { flexGrow: 1, justifyContent: 'center', padding: 24 },
  card: { width: '100%', maxWidth: 460, alignSelf: 'center', padding: 26, backgroundColor: colors.card, borderColor: colors.border, borderWidth: 1, borderRadius: 22 },
  mark: { width: 50, height: 50, borderRadius: 16, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.primary },
  markText: { fontSize: 25 },
  eyebrow: { color: colors.secondary, fontSize: 11, fontWeight: '700', letterSpacing: 2, marginTop: 22 },
  title: { color: colors.text, fontSize: 27, fontWeight: '700', marginTop: 5 },
  subtitle: { color: colors.textMuted, fontSize: 14, lineHeight: 21, marginTop: 7, marginBottom: 20 },
  label: { color: colors.text, fontWeight: '600', fontSize: 13, marginTop: 13, marginBottom: 7 },
  input: { color: colors.text, backgroundColor: colors.background, borderColor: colors.border, borderWidth: 1, borderRadius: 10, paddingHorizontal: 13, paddingVertical: Platform.OS === 'ios' ? 13 : 10, fontSize: 15 },
  message: { color: '#FFB0AD', backgroundColor: 'rgba(255,118,117,0.12)', borderRadius: 9, padding: 11, marginTop: 14, fontSize: 13 },
  success: { color: '#9BE9CF', backgroundColor: 'rgba(0,184,148,0.12)', borderRadius: 9, padding: 11, marginTop: 5, fontSize: 13 },
  submit: { minHeight: 48, backgroundColor: colors.primary, borderRadius: 11, alignItems: 'center', justifyContent: 'center', marginTop: 20 },
  submitText: { color: colors.text, fontSize: 15, fontWeight: '700' },
  disabled: { opacity: 0.65 },
  switch: { color: colors.textMuted, textAlign: 'center', marginTop: 20, fontSize: 13 },
  link: { color: '#A9A0FF', fontWeight: '700' },
});

function apiError(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join(' ');
  if (error?.message?.includes('did not return a session')) return error.message;
  return 'Could not connect to the account service. Check the backend and try again.';
}

export default function AuthScreen({ route, navigation, setUser }) {
  const mode = route.name === 'Signup' ? 'signup' : 'login';
  const registering = mode === 'signup';
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);

  const submit = async () => {
    setError('');
    if (!email.trim() || !password) { setError('Enter your email address and password.'); return; }
    if (registering) {
      if (name.trim().length < 2) { setError('Enter your full name.'); return; }
      if (password !== confirmPassword) { setError('Passwords do not match.'); return; }
      if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{12,}/.test(password)) {
        setError('Use at least 12 characters with uppercase, lowercase, and a number.'); return;
      }
    }
    setPending(true);
    try {
      if (registering) {
        await signup({ name: name.trim(), email: email.trim(), password, confirm_password: confirmPassword });
        navigation.navigate('Login', { notice: 'Account created. Log in to continue.' });
      } else {
        const user = await login(email.trim(), password);
        setUser(user);
      }
    } catch (requestError) {
      setError(apiError(requestError));
    } finally {
      setPending(false);
    }
  };

  return (
    <SafeAreaView style={styles.page}>
      <KeyboardAvoidingView style={styles.page} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          <View style={styles.card}>
            <View style={styles.mark}><Text style={styles.markText}>✉</Text></View>
            <Text style={styles.eyebrow}>INVITATIONSENSE</Text>
            <Text style={styles.title}>{registering ? 'Create your account' : 'Welcome back'}</Text>
            <Text style={styles.subtitle}>{registering ? 'Sign up to start understanding your invitations.' : 'Log in to continue to your invitations.'}</Text>
            {!registering && route.params?.notice ? <Text style={styles.success}>{route.params.notice}</Text> : null}
            {registering ? <>
              <Text style={styles.label}>Full name</Text>
              <TextInput style={styles.input} value={name} onChangeText={setName} autoComplete="name" textContentType="name" placeholder="Your full name" placeholderTextColor={colors.textMuted} maxLength={100} returnKeyType="next" />
            </> : null}
            <Text style={styles.label}>Email address</Text>
            <TextInput style={styles.input} value={email} onChangeText={setEmail} autoCapitalize="none" autoCorrect={false} keyboardType="email-address" autoComplete="email" textContentType="emailAddress" placeholder="you@example.com" placeholderTextColor={colors.textMuted} />
            <Text style={styles.label}>Password</Text>
            <TextInput style={styles.input} value={password} onChangeText={setPassword} secureTextEntry autoComplete={registering ? 'new-password' : 'current-password'} textContentType={registering ? 'newPassword' : 'password'} placeholder={registering ? 'At least 12 characters' : 'Your password'} placeholderTextColor={colors.textMuted} maxLength={72} onSubmitEditing={!registering ? submit : undefined} />
            {registering ? <>
              <Text style={styles.label}>Confirm password</Text>
              <TextInput style={styles.input} value={confirmPassword} onChangeText={setConfirmPassword} secureTextEntry autoComplete="new-password" textContentType="newPassword" placeholder="Enter your password again" placeholderTextColor={colors.textMuted} maxLength={72} />
            </> : null}
            {error ? <Text style={styles.message} accessibilityRole="alert">{error}</Text> : null}
            <TouchableOpacity style={[styles.submit, pending && styles.disabled]} onPress={submit} disabled={pending} accessibilityRole="button">
              {pending ? <ActivityIndicator color={colors.text} /> : <Text style={styles.submitText}>{registering ? 'Sign Up' : 'Log In'}</Text>}
            </TouchableOpacity>
            <Text style={styles.switch}>
              {registering ? 'Already have an account? ' : 'New to InvitationSense? '}
              <Text style={styles.link} onPress={() => navigation.navigate(registering ? 'Login' : 'Signup')}>
                {registering ? 'Log in' : 'Sign up'}
              </Text>
            </Text>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
