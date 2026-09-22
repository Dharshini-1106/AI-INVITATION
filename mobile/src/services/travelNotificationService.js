import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

async function prepareNotifications() {
  if (Platform.OS === 'web') return false;
  const current = await Notifications.getPermissionsAsync();
  const permission = current.status === 'granted'
    ? current
    : await Notifications.requestPermissionsAsync();
  if (permission.status !== 'granted') return false;

  // Android 8+ only presents scheduled notifications on a configured channel.
  await Notifications.setNotificationChannelAsync('travel-reminders', {
    name: 'Travel reminders',
    importance: Notifications.AndroidImportance.HIGH,
    vibrationPattern: [0, 250, 250, 250],
  });
  return true;
}

function futureDate(isoTime) {
  const date = new Date(isoTime);
  return Number.isNaN(date.getTime()) || date <= new Date() ? null : date;
}

/**
 * Schedule separate reminders for this plan. They are never shared between
 * events, so planning one invitation cannot overwrite another invitation.
 */
export async function scheduleTravelNotifications({ plan, event, destination }) {
  if (!plan?.schedule_available) return { scheduled: false, reason: 'Event time is required to schedule a travel reminder.' };
  if (!await prepareNotifications()) return { scheduled: false, reason: 'Notification permission is required for travel reminders.' };

  const venue = destination || plan.destination || event?.venue || event?.address || 'the event venue';
  const eventKey = [event?.event_name, event?.event_type, event?.date, event?.time, venue].filter(Boolean).join('|');
  // Recalculating an event replaces only that event's previous reminders;
  // reminders belonging to other invitation events remain untouched.
  const existing = await Notifications.getAllScheduledNotificationsAsync();
  await Promise.all(existing
    .filter((notification) => notification.content?.data?.travelEventKey === eventKey)
    .map((notification) => Notifications.cancelScheduledNotificationAsync(notification.identifier)));
  const travelTime = plan.travel_duration_text || 'the latest calculated travel time';
  const arrivalTime = new Intl.DateTimeFormat('en-US', {
    hour: 'numeric', minute: '2-digit', timeZone: plan.timezone || event?.timezone || 'Asia/Kolkata',
  }).format(new Date(plan.arrival_time));
  const ids = [];
  const readyAt = futureDate(plan.ready_time);
  const leaveAt = futureDate(plan.departure_time);

  if (readyAt) ids.push(await Notifications.scheduleNotificationAsync({
    content: {
      title: 'Get ready now',
      body: `You should leave home at ${new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZone: plan.timezone || event?.timezone || 'Asia/Kolkata' }).format(new Date(plan.departure_time))} to reach ${venue} before the event.`,
      sound: 'default',
      data: { kind: 'travel-ready', travelEventKey: eventKey, eventName: event?.event_name || event?.event_type || 'Invitation event' },
    },
    trigger: { date: readyAt, channelId: 'travel-reminders' },
  }));
  if (leaveAt) ids.push(await Notifications.scheduleNotificationAsync({
    content: {
      title: 'Time to leave home',
      body: `Leave now to reach ${venue} by ${arrivalTime}. Current travel time: ${travelTime}.`,
      sound: 'default',
      data: { kind: 'travel-leave', travelEventKey: eventKey, eventName: event?.event_name || event?.event_type || 'Invitation event' },
    },
    trigger: { date: leaveAt, channelId: 'travel-reminders' },
  }));
  return { scheduled: ids.length > 0, ids };
}
