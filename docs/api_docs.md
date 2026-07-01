# Internal API Docs

## Authentication
All requests require Bearer token in Authorization header.
Token expires after 24 hours. Refresh using /auth/refresh endpoint.

## Rate Limits
Standard tier: 100 requests/minute.
Premium tier: 1000 requests/minute.
Exceeding limits returns HTTP 429.

## Endpoints
### GET /users
Returns list of all users. Requires admin role.

### POST /users
Creates a new user. Body: {name, email, role}.

### DELETE /users/{id}
Soft-deletes user. Requires admin role.
