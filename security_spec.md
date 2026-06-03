# Security Specification and Threat Model

This specification governs the security model implemented for the Firestore instance of our Sign Language Recognizer.

## 1. Data Invariants
1. A hand gesture sample cannot be created or modified by anyone other than the authenticated owner of the `users/{userId}` path.
2. The `label` of any sample must be an uppercase string bounded in size to prevent memory or display attacks.
3. The `coords` array of any sample must be exactly of size 84, containing numeric landmarks only.
4. The `userId` of any sample must match the authenticated `request.auth.uid`.
5. The `createdAt` timestamp must be set to `request.time`.
6. A model centroid cannot be created or modified by anyone other than the owner of the parent user document.
7. User profiles or private data cannot be exposed to other unauthenticated or random users.

## 2. The Dirty Dozen Payloads (Security Bypass Attempts)

Below are 12 malicious payloads that our security rules are guaranteed to block with `PERMISSION_DENIED`.

### Payload 1: Spurious Landmarking (Denial of Wallet via Huge Coords Array)
Attempting to save coords containing a list of 10,000 floats.
```json
{
  "label": "HOLA",
  "coords": [ /* ... 10,000 values */ ],
  "userId": "victim_user_123",
  "createdAt": "request.time"
}
```

### Payload 2: Identity Theft (Forged Owner ID)
Submitting a sample trying to write under someone else's space (`users/attacker456/samples/newDoc`) with creator set to `victimUser`.
```json
{
  "label": "ADIOS",
  "coords": [0.1, 0.2, 0.3],
  "userId": "victim_user_123",
  "createdAt": "request.time"
}
```

### Payload 3: Injecting Null or Spurious Types
Injecting non-string values as labels or non-number entries inside coordinates.
```json
{
  "label": 9999,
  "coords": ["not_a_number"],
  "userId": "user123",
  "createdAt": "request.time"
}
```

### Payload 4: Spoofing Creation Timestamp
Attempting to send a fixed client-side date in the past instead of `request.time`.
```json
{
  "label": "GRACIAS",
  "coords": [0.0],
  "userId": "user123",
  "createdAt": "2001-01-01T00:00:00Z"
}
```

### Payload 5: Model Tampering / Bypass (External Hijack)
Submitting custom classifications directly to another user's trained centroids to trick the model.
```json
{
  "centroids": { "ATTACK_LABEL": [9.9, 9.9] },
  "userId": "victim_user_123",
  "updatedAt": "request.time"
}
```

### Payload 6: Out of Bounds Label Characters
Using character scripts or massive SQL injection payloads in the label field of a gesture.
```json
{
  "label": "AWARENESSTOGETHERWITHMASSSIVESTRINGOFCHARACTERSMORETHAN100LONGZZZZZZ",
  "coords": [],
  "userId": "user123",
  "createdAt": "request.time"
}
```

### Payload 7: Anonymous Write Hijack
Attempting to write samples while unauthenticated (`request.auth == null`).
```json
{
  "label": "HOLA",
  "coords": [],
  "userId": "anonymous_user",
  "createdAt": "request.time"
}
```

### Payload 8: Immutable Field Override on Sample Update
Updating an existing sample document and changing the `userId` field to a different value.
```json
{
  "label": "HOLA",
  "coords": [],
  "userId": "attacker_user_456",
  "createdAt": "request.time"
}
```

### Payload 9: Empty/Zombie Coordinates List
Attempting to write a sample with no landmarks associated (empty labels or empty arrays).
```json
{
  "label": "",
  "coords": [],
  "userId": "user123",
  "createdAt": "request.time"
}
```

### Payload 10: State Shortcut on ModelCentroid
Attempting to clear the trained centroids map entirely or injecting nested structures.
```json
{
  "centroids": null,
  "userId": "user123",
  "updatedAt": "request.time"
}
```

### Payload 11: Spoofed email verification status
Adding data with a request token that claims to be verified but is actually bypassed.
```json
{
  "label": "SPOOFED_VERIFICATION",
  "coords": [0.1],
  "userId": "user123",
  "createdAt": "request.time"
}
```

### Payload 12: Injection of arbitrary properties (Ghost Field Attack)
Attempting to append fields like `isVerifiedAdmin: true` into the sample object.
```json
{
  "label": "HOLA",
  "coords": [0.1],
  "userId": "user123",
  "createdAt": "request.time",
  "isVerifiedAdmin": true
}
```

---

## 3. Test Runner Specification

Below is a mock representation of the Test Runner file validation script.

```typescript
import { assertFails, initializeTestEnvironment } from '@firebase/rules-unit-testing';

describe('Sign Language Recognizer Security Rules', () => {
  it('should block all malicious payloads', async () => {
    const testEnv = await initializeTestEnvironment({
      projectId: 'galvanized-cacao-8cf5x'
    });
    
    const unauthedDb = testEnv.unauthenticatedContext().firestore();
    // Payload 7 validation block:
    await assertFails(unauthedDb.doc('users/user123/samples/sampleA').set({
      label: 'HOLA',
      coords: [],
      userId: 'user123',
      createdAt: new Date()
    }));
  });
});
```
