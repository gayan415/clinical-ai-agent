# CardioMEMS HF System — Hemodynamic Monitoring Protocol

## Overview
The CardioMEMS HF System is an implantable pulmonary artery (PA) pressure sensor manufactured by Abbott. It is a paperclip-sized wireless device that transmits daily PA pressure readings to clinicians, enabling proactive heart failure management. The system detects hemodynamic congestion approximately 3 weeks before symptoms appear.

## How It Works
1. **Implantation:** Sensor is deployed in a distal branch of the pulmonary artery via right heart catheterization (outpatient procedure, ~20 minutes)
2. **Daily Readings:** Patient lies on an electronic pillow at home and transmits PA pressure readings wirelessly
3. **Data Review:** Clinicians review PA pressure trends through the Merlin.net Patient Care Network (PCN) HF Portal
4. **Medication Adjustment:** Clinicians adjust diuretics and vasodilators based on pressure trends, preventing decompensation

## PA Pressure Reference Ranges
- **Normal PA systolic:** 15-30 mmHg
- **Normal PA diastolic:** 4-12 mmHg
- **Normal PA mean:** 9-18 mmHg
- **Elevated PA diastolic (congestion signal):** > 20 mmHg
- **Critical elevation:** PA diastolic > 25 mmHg — requires urgent clinical intervention

## Clinical Response Protocol
| PA Diastolic Trend | Clinical Action |
|---|---|
| Stable within normal range | Continue current therapy, routine follow-up |
| Rising trend (15 → 20 mmHg over 5-7 days) | Increase diuretic dose, schedule follow-up in 3-5 days |
| Elevated (20-25 mmHg) | Aggressive diuretic titration, consider adding vasodilator, follow-up in 1-3 days |
| Critical (> 25 mmHg or acute rise > 5 mmHg/day) | Urgent clinical contact, consider IV diuretics, evaluate for hospitalization |
| Dropping below normal (< 8 mmHg) | Reduce diuretics to avoid dehydration and hypotension |

## Evidence Base
- **CHAMPION Trial:** 28% reduction in heart failure hospitalizations at 6 months for NYHA Class III patients
- **GUIDE-HF Trial:** Demonstrated benefit across broader heart failure population
- **Meta-analysis (HFrEF):** 25% mortality reduction at 2 years
- Cost-effective at approximately $17,000 per quality-adjusted life year (QALY)

## FDA Indications
- **Original (2014):** NYHA Class III heart failure with a heart failure hospitalization in the prior 12 months
- **Expanded (February 2022):** NYHA Class II heart failure with elevated natriuretic peptides (BNP ≥ 250 pg/mL or NT-proBNP ≥ 1000 pg/mL)
- Estimated 1.2 million additional eligible patients in the US following expanded indication

## Integration with AI/ML
- Daily PA pressure readings create a rich longitudinal dataset for predictive modeling
- Machine learning can identify subtle pressure patterns that predict decompensation earlier than rule-based thresholds
- Risk stratification models can prioritize which patients need immediate clinician review
- Automated triage could reduce clinician workload while improving patient outcomes
