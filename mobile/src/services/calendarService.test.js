jest.mock('expo-calendar', () => ({}));

import { parseDate, parseTime, reminderAlarms } from './calendarService';

const formatDate = (date) => [
  date.getFullYear(),
  String(date.getMonth() + 1).padStart(2, '0'),
  String(date.getDate()).padStart(2, '0'),
].join('-');

test.each([
  ['11th February 2026', '2026-02-11'],
  ['11 February 2026', '2026-02-11'],
  ['11th Feb 2026', '2026-02-11'],
  ['11 Feb 2026', '2026-02-11'],
  ['February 11, 2026', '2026-02-11'],
  ['11/02/2026', '2026-02-11'],
  ['11-02-2026', '2026-02-11'],
  ['11.02.2026', '2026-02-11'],
  ['2026-02-11', '2026-02-11'],
  ['21st February 2026', '2026-02-21'],
  ['3rd March 2026', '2026-03-03'],
])('parses %s as %s', (input, expected) => {
  expect(formatDate(parseDate(input))).toBe(expected);
});

test.each([
  ['5.00 PM', { hours: 17, minutes: 0 }],
  ['5:00 PM', { hours: 17, minutes: 0 }],
  ['5 PM', { hours: 17, minutes: 0 }],
  ['17:00', { hours: 17, minutes: 0 }],
])('parses %s', (input, expected) => {
  expect(parseTime(input)).toEqual(expected);
});

test('adds only future calendar reminder offsets', () => {
  const start = new Date(2026, 1, 11, 17, 0, 0);
  expect(reminderAlarms(start, new Date(2026, 1, 8, 12, 0, 0))).toEqual([
    { relativeOffset: -2880 },
    { relativeOffset: -1440 },
    { relativeOffset: -60 },
  ]);
  expect(reminderAlarms(start, new Date(2026, 1, 11, 15, 30, 0))).toEqual([
    { relativeOffset: -60 },
  ]);
  expect(reminderAlarms(start, new Date(2026, 1, 11, 17, 0, 0))).toEqual([]);
});
