"""Focused persistence and ownership tests for saved events."""
from bson import ObjectId
import pytest
from fastapi import HTTPException

from app.api.routes import events
from app.api.routes.events import EventPayload, SchedulePayload, TravelPlanPayload


class FakeCursor(list):
    def sort(self, key, direction):
        return FakeCursor(sorted(self, key=lambda item: item.get(key), reverse=direction < 0))


class FakeCollection:
    def __init__(self):
        self.documents = []

    def insert_one(self, document):
        saved = dict(document, _id=ObjectId())
        self.documents.append(saved)
        return type("InsertResult", (), {"inserted_id": saved["_id"]})()

    def find(self, query):
        return FakeCursor(document for document in self.documents if all(
            document.get(key) == value for key, value in query.items()
        ))

    def find_one(self, query):
        return next((document for document in self.documents if all(
            document.get(key) == value for key, value in query.items()
        )), None)

    def update_one(self, query, update):
        document = self.find_one(query)
        if not document:
            return type("UpdateResult", (), {"matched_count": 0})()
        document.update(update["$set"])
        return type("UpdateResult", (), {"matched_count": 1})()

    def delete_one(self, query):
        document = self.find_one(query)
        if document:
            self.documents.remove(document)
        return type("DeleteResult", (), {"deleted_count": int(document is not None)})()


class FakeDatabase:
    def __init__(self):
        self.events = FakeCollection()


@pytest.fixture
def database(monkeypatch):
    fake = FakeDatabase()
    monkeypatch.setattr(events, "database", lambda: fake)
    return fake


def test_event_travel_and_schedule_persist_for_owner(database):
    user = {"id": str(ObjectId())}
    created = events.create_event(
        EventPayload(event={"event_name": "Wedding", "venue": "City Hall"}), user
    )

    plan = events.save_travel_plan(
        created["id"],
        TravelPlanPayload(
            travel_plan={"travel_mode": "DRIVE", "departure_time": "2026-10-02T17:00:00+05:30"},
            request={"origin": {"latitude": 9.1, "longitude": 77.5}, "preparation_minutes": 60},
        ),
        user,
    )
    updated = events.update_event(
        created["id"], EventPayload(event={"event_name": "Updated wedding"}), user
    )
    scheduled = events.save_schedule(
        created["id"], SchedulePayload(schedule={"status": "scheduled", "calendar_event_ids": ["local-id"]}), user
    )

    assert plan["travel_plan"]["travel_mode"] == "DRIVE"
    assert plan["travel_plan"]["request"]["preparation_minutes"] == 60
    assert updated["event"]["event_name"] == "Updated wedding"
    assert updated["travel_plan"]["travel_mode"] == "DRIVE"
    assert scheduled["schedule"]["calendar_event_ids"] == ["local-id"]
    assert events.get_event(created["id"], user)["id"] == created["id"]
    assert len(events.list_events(user)) == 1


def test_event_id_cannot_be_used_by_another_user(database):
    owner = {"id": str(ObjectId())}
    other_user = {"id": str(ObjectId())}
    created = events.create_event(EventPayload(event={"event_name": "Private"}), owner)

    for action in (
        lambda: events.get_event(created["id"], other_user),
        lambda: events.update_event(created["id"], EventPayload(event={}), other_user),
        lambda: events.save_travel_plan(created["id"], TravelPlanPayload(travel_plan={}), other_user),
        lambda: events.save_schedule(created["id"], SchedulePayload(schedule={}), other_user),
        lambda: events.delete_event(created["id"], other_user),
    ):
        with pytest.raises(HTTPException) as error:
            action()
        assert error.value.status_code == 404

    assert events.get_event(created["id"], owner)["event"]["event_name"] == "Private"


def test_list_events_is_scoped_to_authenticated_user(database):
    owner = {"id": str(ObjectId())}
    other_user = {"id": str(ObjectId())}
    events.create_event(EventPayload(event={"event_name": "Owner event"}), owner)
    events.create_event(EventPayload(event={"event_name": "Other event"}), other_user)

    assert [item["event"]["event_name"] for item in events.list_events(owner)] == ["Owner event"]


def test_saved_events_are_restored_for_same_user_after_new_session(database):
    user_id = str(ObjectId())
    events.create_event(EventPayload(event={"event_name": "Reception"}), {"id": user_id})

    # Auth sessions are replaceable; persisted records remain linked to the stable user ID.
    restored_session_user = {"id": user_id}
    assert events.list_events(restored_session_user)[0]["event"]["event_name"] == "Reception"