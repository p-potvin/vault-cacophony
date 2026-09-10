# Magpie voices + diarization: multi-speaker and multilingual

nemotron-3.5 `--language auto --diarize`, sortformer 4-speaker diarizer.
All audio is Magpie TTS with baked voices, spliced with 350 ms gaps.

## 4 speakers, no Aria (CONTROL) — `convctl`

- true speakers **4**, diarizer emitted **4** labels
- word-level speaker accuracy **96.2%** (128/133)
- languages detected: `['en-US']`

| time | true speaker | assigned | text |
|---|---|---|---|
| 00:00 | John | John | Alright, let's start. Jason, where are we on the migration? |
| 00:05 | Jason | Jason | Two of the four services are cut over. The billing one is still on the |
| 00:10 | Sofia | Sofia | Billing is the one I'm worried about. It has the most downstream consu |
| 00:15 | Leo | Leo | I can take billing this week if nobody else has capacity for it. |
| 00:19 | John | John | That would help. Jason, would you pair with Leo on the first day? |
| 00:24 | Jason | Jason | Sure. Most of the work is in the retry logic, and I already mapped tha |
| 00:29 | Sofia | Sofia | Please write down the rollback steps before you start, not after. |
| 00:33 | Leo | Leo | Understood. I will have a document ready before I touch anything in pr |
| 00:38 | John | John | Good. Let's reconvene Thursday and see where the billing service stand |
| 00:44 | Sofia | Sofia | Works for me. I will bring the consumer list so we know who to notify. |

**Transcript:** All right, let's start Jason. Where are we on the migration? Two of the four services are cut over the billing one is still on the old cluster. Billing is the one I'm worried about. It has the most downstream consumers. I can take billing this week if nobody else has capacity for it. That would help. Jason would you pair with Leo on the first day? Sure, most of the work is in the retry logic, and I already mapped that out. Please write down the rollback steps before you start not after understood I will have a document ready before I touch anything in production. Good. Let's reconvene Thursday and see where the billing service stands works for me. I will bring the consumer list so we know who to notify.

---

## 4 speakers incl. Aria — `conv4`

- true speakers **4**, diarizer emitted **3** labels
- word-level speaker accuracy **66.9%** (89/133)
- languages detected: `['en-US']`

| time | true speaker | assigned | text |
|---|---|---|---|
| 00:00 | John | John | Alright, let's start. Aria, where are we on the migration? |
| 00:04 | Aria | Sofia **MISS** | Two of the four services are cut over. The billing one is still on the |
| 00:08 | Sofia | Sofia | Billing is the one I'm worried about. It has the most downstream consu |
| 00:13 | Jason | Sofia **MISS** | I can take billing this week if nobody else has capacity for it. |
| 00:17 | John | John | That would help. Aria, would you pair with Jason on the first day? |
| 00:22 | Aria | Sofia **MISS** | Sure. Most of the work is in the retry logic, and I already mapped tha |
| 00:26 | Sofia | Sofia | Please write down the rollback steps before you start, not after. |
| 00:30 | Jason | Jason | Understood. I will have a document ready before I touch anything in pr |
| 00:35 | John | John | Good. Let's reconvene Thursday and see where the billing service stand |
| 00:40 | Sofia | Sofia | Works for me. I will bring the consumer list so we know who to notify. |

**Transcript:** All right, let's start Arya where are we on the migration? Two of the four services are cut over the billing one is still on the old cluster. Billing is the one I'm worried about. It has the most downstream consumers. I can take billing this week if nobody else has capacity for it that would help Arya would you pair with Jason on the first day sure most of the work is in the retry logic and I already mapped that out. Please write down the rollback steps before you start, not after understood. I will have a document ready before I touch anything in production good let's reconvene Thursday and see where the billing service stands works for me I will bring the consumer list so we know who to notify.

---

## 5 speakers — `conv5`

- true speakers **5**, diarizer emitted **4** labels
- word-level speaker accuracy **67.0%** (69/103)
- languages detected: `['en-US']`

| time | true speaker | assigned | text |
|---|---|---|---|
| 00:00 | John | John | We have five people today, so let's keep answers short. |
| 00:04 | Sofia | Aria **MISS** | Revenue is tracking to plan. Nothing surprising this month. |
| 00:07 | Aria | Aria | Engineering is blocked on the vendor contract, otherwise on schedule. |
| 00:11 | Jason | Jason | Support volume dropped after the onboarding fix went out. |
| 00:15 | Leo | Jason **MISS** | Marketing wants to know when we can announce the new tier. |
| 00:19 | John | John | Leo, hold the announcement until the contract clears. |
| 00:24 | Aria | Aria | That should be next week if legal responds by Friday. |
| 00:27 | Leo | Leo | Then I will draft it now and sit on it until you give the word. |
| 00:32 | Sofia | Aria **MISS** | Make sure the pricing page matches whatever the draft claims. |
| 00:35 | Jason | Jason | And send me the copy early so support can prepare answers. |

**Transcript:** We have five people today, so let's keep answers short. Revenue is tracking to plan nothing surprising this month. Engineering is blocked on the vendor contract otherwise on schedule. Support volume dropped after the onboarding fix went out. Marketing wants to know when we can announce the new tier. Leo. hold the announcement until the contract clears. That should be next week if legal responds by Friday. Then I will draft it now and sit on it until you give the word. Make sure the pricing page matches whatever the draft claims, and send me the copy early so support can prepare answers.

---

## 3 speakers, en/es/fr — `convml`

- true speakers **3**, diarizer emitted **2** labels
- word-level speaker accuracy **71.3%** (67/94)
- languages detected: `['en-US', 'es-US', 'fr-FR']`

| time | true speaker | assigned | text |
|---|---|---|---|
| 00:00 | John | John | Thanks for joining. I know we have people from three offices today. |
| 00:04 | Sofia | Sofia | Claro. Desde Madrid todo listo, ya revisamos los números del trimestre |
| 00:10 | Aria | Sofia **MISS** | Bonjour à tous. De notre côté, le rapport sera prêt jeudi matin. |
| 00:15 | John | John | Perfect. Sofia, can you summarise the numbers for everyone? |
| 00:20 | Sofia | Sofia | Sí, los ingresos subieron un doce por ciento pero los costes también. |
| 00:25 | Aria | Sofia **MISS** | C'est la même tendance chez nous, surtout sur le support technique. |
| 00:29 | John | John | So it is consistent across regions. That is actually useful to know. |
| 00:35 | Sofia | Sofia | Exacto. Propongo que revisemos el proceso antes de contratar a nadie. |
| 00:40 | Aria | Sofia **MISS** | Je suis d'accord. Recruter maintenant ne réglerait rien du tout. |

**Transcript:** Thanks for joining. I know we have people from three offices today. Claro, desde Madrid todo listo, ya revisamos los números del trimestre. Bonjour a tus côté, le rapport sera prêt jeudi matin perfect Sophia, can you summarize the numbers for everyone? See, los ingresos subieron un doce por ciento pero los costes también será la même tendance chez nous, surtout sur le support technique, so it is consistent across regions that is actually useful to know exacto propongo que revisemos el proceso antes de contratar a nadie recrutez maintenant ne réglerez rien du tout.

---

