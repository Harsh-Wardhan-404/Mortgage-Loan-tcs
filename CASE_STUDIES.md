# Case Studies for Model Testing

This document contains case studies to test the regulatory compliance Q&A model.

## Case Study 1: Capital Adequacy Compliance

**Scenario:** A housing finance company (HFC) is preparing for its annual regulatory audit. The compliance team needs to understand the capital adequacy requirements to ensure they meet all regulatory standards.

**Question:** What are the capital adequacy requirements for housing finance companies?

**Expected Domain:** Capital Adequacy, Regulatory Compliance

**Evaluation Criteria:**
- Should mention specific capital adequacy ratios
- Should reference regulatory framework (RBI/NHB)
- Should provide actionable compliance information

---

## Case Study 2: KYC Update for Minor Account

**Scenario:** A customer opened a housing finance account when they were 17 years old. They are now turning 18 and the bank needs to update their KYC documentation.

**Question:** What is required for KYC updates when a minor becomes an adult?

**Expected Domain:** KYC/AML, Customer Due Diligence

**Evaluation Criteria:**
- Should mention fresh photographs
- Should mention CDD (Customer Due Diligence) documents
- Should reference current standards

---

## Case Study 3: Exposure Limit Compliance

**Scenario:** A State Cooperative Bank (StCB) wants to expand its housing finance portfolio. The risk management team needs to verify the maximum exposure limits allowed by regulations.

**Question:** What is the total exposure limit to housing finance and CRE-Residential Housing for all StCBs/DCCBs?

**Expected Domain:** Risk Management, Exposure Limits

**Evaluation Criteria:**
- Should provide specific percentage (5% of total assets)
- Should mention StCBs/DCCBs specifically
- Should reference regulatory limits

---

## Case Study 4: Registration Cancellation

**Scenario:** A housing finance institution has been non-compliant with several regulatory requirements. The Reserve Bank is considering cancellation of their registration certificate.

**Question:** Under what conditions can the Reserve Bank cancel a certificate of registration for a housing finance institution?

**Expected Domain:** Registration, Regulatory Enforcement

**Evaluation Criteria:**
- Should list specific conditions for cancellation
- Should mention non-compliance scenarios
- Should reference Reserve Bank powers

---

## Case Study 5: Securities Market Value

**Scenario:** A non-banking financial company needs to maintain securities for public deposits. The compliance officer needs to understand the market value requirements.

**Question:** What is the market value requirement for securities held for depositors?

**Expected Domain:** Securities, Deposit Protection

**Evaluation Criteria:**
- Should mention market value requirements
- Should reference percentage of public deposits
- Should mention Chapter III of directions

---

## Case Study 6: Board Regulations

**Scenario:** A housing finance company is updating its board regulations and needs to understand what matters can be covered in the regulations.

**Question:** What matters can the regulations provided by the Board cover?

**Expected Domain:** Corporate Governance, Board Regulations

**Evaluation Criteria:**
- Should list various matters (fees, allowances, etc.)
- Should mention director elections
- Should reference regulatory framework

---

## Case Study 7: Maturity Profile Categorization

**Scenario:** A financial institution needs to prepare its maturity profile report and needs to understand how to categorize different types of capital and borrowings.

**Question:** How are bank borrowings in the nature of WCDL, CC categorized in the maturity profile?

**Expected Domain:** Asset-Liability Management, Maturity Profile

**Evaluation Criteria:**
- Should mention specific time-bucket
- Should reference WCDL and CC
- Should provide categorization guidance

---

## Testing Instructions

1. **Load the Model**: Use the frontend application to load the fine-tuned model
2. **Select Case Study**: Choose a case study from the dropdown
3. **Generate Answer**: Click "Get Answer" to see the model's response
4. **Evaluate**: Rate the answer on:
   - Relevance to the question
   - Regulatory accuracy
   - Completeness of information
5. **Compare**: Compare with expected answers from training data

## Expected Model Behavior

The model should:
- Provide accurate regulatory information
- Use proper regulatory terminology
- Reference specific sections or percentages when available
- Give actionable compliance guidance
- Maintain professional tone

## Success Criteria

A successful test should show:
- ✅ Model provides relevant answers
- ✅ Answers contain accurate regulatory information
- ✅ Model uses domain-specific terminology
- ✅ Answers are complete and actionable
- ✅ Model demonstrates understanding of regulatory context


