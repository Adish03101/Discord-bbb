## **Two problems to solve**

1. **Sense of completion** even in multi-round pipelines
2. **Exclusivity signal** for CEO / Senior mascot

| **Constraint**                                                                            | **Why it’s tricky**                                                                                                                         | **Pragmatic UX answer**                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Tease CEO / Sr. mascots even when they’re not valid for the current interview type** | Personality-picker appears _after_ the user has already chosen an interview type, so a “CEO” face feels out of place for Generic Screening. | **Inline-tease, but context-aware**: show the CEO tile _dimmed_ + lock + thin caption: “Only active in Boardroom-level interviews.”• User sees it, understands exclusivity.• No logical break—the UI itself explains why it’s disabled.     |
| **B. Preserve a “completed” feeling despite multi-job pipelines**                         | Same user may track multiple jobs; generic practise sessions shouldn’t feel half-done.                                                      | Dual progress system:1. **Per-Job Ring** on each Job Card (circular stage tracker).2. **Global XP Ribbon** in header (total interviews done).This way, any single session → instant XP & badge, while job-rings give longer-arc completion. |
|                                                                                           |                                                                                                                                             |                                                                                                                                                                                                                                             |



## **1. Expanded “Interview Situation” Library (Marketing-focused)**

|**Situation Cluster**|**Typical Prompts in a Real Marketing Interview**|**Why It Matters**|
|---|---|---|
|**Brand Case Study**|“Redesign X brand’s positioning for Gen Z.”|Tests strategic thinking + storytelling.|
|**GTM Plan**|“Launch a new feature in 30 days; outline channels, budget, KPIs.”|Cross-functional & executional skills.|
|**Data Interpretation**|“Here’s last quarter’s funnel—what’s the bottleneck?”|Analytical rigor, narrative with numbers.|
|**Creative Critique**|“Evaluate this campaign: what works/what fails?”|Expression, design sensitivity, courage to iterate.|
|**Stakeholder Alignment**|“CMO pushes for virality; CFO wants CAC ↓ — what’s your trade-off?”|Negotiation, mindset, diplomacy.|
|**Salary/Role Negotiation**|“Why 40 LPA and Director title?”|Self-worth, composure, persuasion.|
|**Curve-ball / Wildcard**|“TikTok banned tomorrow—plan B?”|Agility, lateral thinking.|

## **2. Four tools Overlay — Example Mapping for “Brand Case Study”**


| **Mindset**          | **Skills**                  | **Expression**                     | **Story**                        |
| -------------------- | --------------------------- | ---------------------------------- | -------------------------------- |
| Confidence in vision | Framework use (3C/4P, etc.) | Clear, audience-tuned language     | Weave narrative of brand rebirth |
| Customer-first bias  | Research synthesis          | Visual props → slides / whiteboard | Show past wins → future impact   |

## **3. Structural Options**

|**Model**|**User-side Feel**|**Completion Signal (first session)**|**Depth / Retention**|**Paywall Hooks**|
|---|---|---|---|---|
|**A. Classic “Rounds” (3-step)**|Looks like a real hiring pipeline (Screen → Manager → CEO)|✅ “Round 1 Complete” badge|High (forced journey)|Seniority mascots unlocked Round 3|
|**B. Scenario-Pick (no fixed rounds)**|Quiz-like: pick any situation immediately|✅ Green check on each scenario finished|Medium (user might cherry-pick, lower stickiness)|Premium scenarios (Negotiation, CEO)|
|**C. Hybrid**_(Recommended starting point)_|Session 1 ⇒ one _Core Round_ + user-picked _Bonus Scenario_|✅ Core complete bar + optional “continue” CTA|High (reward + next-step nudge)|Bonus Scenarios + Senior Mascot skins|





# **Encase – Interview Rounds & Mascot Mapping**

  

_Version 0.1 | Draft for team review_

---

## **1  Purpose & Scope**

  

This document captures the latest decisions, open questions, and design logic behind moving to a **round-based interview system** while preserving **seniority mascots** and a strong **sense of completion** for every user segment.

---

## **2  Problems We’re Solving**

| **ID** | **Problem Statement**                                                                                | **Why it Matters**                 |
| ------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------- |
| P-1    | **Exclusivity Signal** – Users must _notice_ CEO / Sr. Manager personas early and aspire to them.    | Drives emotional stakes & upsell.  |
| P-2    | **Sense of Completion** – Even in multi-round, multi-job journeys, each session should feel “done”.  | Retains first-time & casual users. |
| P-3    | **Persona Differentiation** – CEO & Sr. Manager rounds can’t feel like “same dialogue with new hat”. | Upholds realism & learning value.  |

---

## **3  Key Decisions (TL;DR)**

1. **Round Framework**: 3 core tiers – _Screening_, _Manager_, _Boardroom_.
    
2. **Locked Seniority Mascots**: Visible from day 0, selectable only after unlock rule:
    
    - ✅ Auto-unlock after completing _N_ interviews 
        
    
3. **Dual Progress System**
    
    - _Global XP ribbon_ (all interviews)
        
    - _Per-Job stage ring_ (specific pipelines)
        
    
4. **Gamification Hooks**: Badges, cameo pop-ups, email nudges.
    

---

## **4  User Flow Overview**

```
Dashboard
 ├── Job Cards (each with Stage Ring)
 │     └── “Take Interview” → Interview-Selection
 └── Practice (no job) → Interview-Selection
           └── Personality-Picker
```

### **4.1  User Segments & What They Need**

| **Segment** | **First Win**            | **Long-Term Driver**    |
| ----------- | ------------------------ | ----------------------- |
| First-time  | Immediate feedback badge | See locked CEO card     |
| Returning   | New scenario unlocked    | XP ribbon growth        |
| Power-user  | Senior rounds            | Skill token progression |

---

## **5  Interview-Round Framework**

|**Round**|**Trigger Condition**|**Persona Availability**|**Completion Artifact**|
|---|---|---|---|
|**R1 Screening**|Always|Gentle / Tough coach|Badge + PDF|
|**R2 Manager**|Complete ≥1 R1|+ Sr. Manager (locked)|Badge + Skill Token|
|**R3 Boardroom**|Complete ≥3 total|CEO (locked)|Badge + Skill Token + Cameo feedback|

> **Unlock Logic**: if completed_interviews >= threshold OR user_is_pro.

---

## **6  Mascot Taxonomy & Gating**

|**Tier**|**Skin**|**Voice Cue**|**Unlock**|
|---|---|---|---|
|Base|Gentle Coach|Friendly mentor|Free|
|Base|Tough Coach|Direct, high-tempo|Free|
|Senior|Sr. Manager – Analytical|Data-first|R2 / Pro|
|Senior|CEO – Visionary|Big-picture|R3 / Pro|
|Senior|CEO – Challenger|Salary pushback|R3 / Pro|

Visual: single base rig + accessory layers (cost efficient).

---

## **7  Emotional-Stake Mechanics**

1. **Locked Tiles** – Dimmed with tooltip “Boardroom-only persona”.
    
2. **Progress Ribbon** – Persistent, header: “40 % to Boardroom access”.
    
3. **Cameo Hook** – Post-R1 popup silhouette: “Meet me after two more wins.”
    
4. **Per-Round Badges** – Instant confetti + downloadable PDF.
    

---

## **8  Training Tools Overlay (Four Minds)**

  

Example: _Brand Case Study_ (Marketing)

|**Mindset**|**Skills**|**Expression**|**Story**|
|---|---|---|---|
|Confidence|Framework use|Clear narrative|Brand rebirth arc|

_Engine auto-maps feedback to drills._

---

## **9  Open Design Questions**

|**Q**|**Topic**|**Notes**|
|---|---|---|
|1|Tooltip tone|Playful vs formal?|
|2|Free-tier interview limits|Current: 3/day – confirm?|
|3|Persona art style|Separate rigs vs accessory layers?|

---

## **10  Next Steps & Owners**

|**Task**|**Owner**|**ETA**|
|---|---|---|
|Finalise unlock thresholds|Product (Yash)|DD MMM|
|Wireframing personality picker|Design (Deep)|DD MMM|
|XP ribbon backend logic|Eng (Jai)|DD MMM|
|Copy & localisation|Content (Navdeep)|DD MMM|

---

## **11  Appendix**

- **Glossary**: R1/R2/R3, XP ribbon, Skill Token…
    
- **Reference Links**: Figma #123, Doc “Interview Scenarios v0.3”.