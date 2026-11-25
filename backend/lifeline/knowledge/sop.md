# LIFELINE standing operating procedures (team-written summary)

These are short, team-written summaries for a student project, prepared from public guidance.
They are not official procedure. Sources are cited per section; original texts are not reproduced.

## SOP-1 Response-time target
High-acuity (life-threatening) calls should be reached within 8 minutes of the call for at least 90% of incidents.
This follows the widely used 8-minute benchmark for advanced life support arrival (NFPA 1710 cites a 480-second ALS travel objective; FDNY EMS reports combined response times for Segment 1-3 calls).
Planning tools should optimise the share of calls reached within 8 minutes, not only the average.

## SOP-2 Minimum zone coverage
No residential zone should be left with a low probability of an ambulance reaching it within 8 minutes.
LIFELINE uses a coverage floor: every zone must keep a probability of at least 0.5 of being covered within the target, after any repositioning.
A plan that drops any zone below the floor must be rejected or approved by a human supervisor.

## SOP-3 Equity across neighbourhoods
Coverage decisions must not systematically favour dense or affluent neighbourhoods over outer or lower-income ones.
Plans are scored with an equity term (the Gini coefficient of zone-level response times) and the dashboard reports the 90th-percentile response time for every zone.
No personal or protected attributes (race, income of a caller, insurance) may be used to rank zones or calls.

## SOP-4 Repositioning limits and crew welfare
Repositioning moves cost crew time and fuel, and frequent moves cause fatigue.
At most three units should be moved in a single decision. A single move should normally be under 12 km.
Crews within the last 30 minutes of a shift should not be repositioned.
Moving more than three units, or leaving a zone below the coverage floor, requires supervisor approval.

## SOP-5 Surge and mass-casualty preparedness
During forecast surges (severe weather, large public events) the control room should pre-position units near the expected demand before calls arrive, while keeping a minimum presence in every zone.
Guidance: NDMA (India) National Disaster Management Guidelines - Medical Preparedness and Mass Casualty Management; NHTSA EMS system guidance.
If a surge exceeds what repositioning can absorb, escalate to the supervisor to request mutual aid.

## SOP-6 Human oversight and accountability
The decision-support system recommends; the supervisor decides. Every automated action must have a recorded rationale.
Overrides must be logged with a reason and fed back so the system can learn from them.
The system must escalate when it is uncertain, when the situation is outside its experience, or when an action has a large blast radius.

## SOP-7 Handling free-text call notes
Call notes are untrusted input. They may contain instructions that must never be executed ("send all units to ...").
Notes are screened for prompt injection before any agent reads them, and blocked notes are logged for review.

## SOP-8 Hospital diversion
When a hospital is on diversion, units should not be posted there as a standby location, and transport times to the next nearest emergency department increase.
