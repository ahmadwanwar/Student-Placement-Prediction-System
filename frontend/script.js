// Talk to the Flask API on the same origin (the Flask app also serves these files).
const API_BASE = "";

const fieldsEl = document.getElementById("fields");
const formEl = document.getElementById("predict-form");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");
const predictionEl = document.getElementById("result-prediction");
const probabilityEl = document.getElementById("result-probability");
const barsEl = document.getElementById("result-bars");
const submitBtn = document.getElementById("submit-btn");
const sampleBtn = document.getElementById("sample-btn");

let SCHEMA = null;

// Reasonable display labels (fallback to humanised column name).
const LABELS = {
  age: "Age",
  gender: "Gender",
  cgpa: "CGPA",
  branch: "Branch",
  college_tier: "College Tier",
  internships_count: "Internships",
  projects_count: "Projects",
  certifications_count: "Certifications",
  coding_skill_score: "Coding Skill Score",
  aptitude_score: "Aptitude Score",
  communication_skill_score: "Communication Score",
  logical_reasoning_score: "Logical Reasoning Score",
  hackathons_participated: "Hackathons Participated",
  github_repos: "GitHub Repos",
  linkedin_connections: "LinkedIn Connections",
  mock_interview_score: "Mock Interview Score",
  attendance_percentage: "Attendance %",
  backlogs: "Backlogs",
  extracurricular_score: "Extracurricular Score",
  leadership_score: "Leadership Score",
  volunteer_experience: "Volunteer Experience",
  sleep_hours: "Sleep Hours",
  study_hours_per_day: "Study Hours / Day",
};

const SAMPLE = {
  age: 21,
  gender: "Male",
  cgpa: 8.2,
  branch: "CSE",
  college_tier: "Tier 1",
  internships_count: 2,
  projects_count: 4,
  certifications_count: 3,
  coding_skill_score: 85,
  aptitude_score: 78,
  communication_skill_score: 80,
  logical_reasoning_score: 82,
  hackathons_participated: 2,
  github_repos: 6,
  linkedin_connections: 500,
  mock_interview_score: 78,
  attendance_percentage: 90,
  backlogs: 0,
  extracurricular_score: 70,
  leadership_score: 65,
  volunteer_experience: "Yes",
  sleep_hours: 7,
  study_hours_per_day: 4,
};

function humanize(name) {
  return name
    .split("_")
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(" ");
}

function buildField(col) {
  const wrap = document.createElement("div");
  wrap.className = "field";

  const label = document.createElement("label");
  label.htmlFor = `f-${col}`;
  label.textContent = LABELS[col] || humanize(col);
  wrap.appendChild(label);

  const rule = (SCHEMA.validation_rules || {})[col];

  let input;
  if (SCHEMA.categorical_cols.includes(col)) {
    input = document.createElement("select");
    const options = SCHEMA.categorical_options[col] || [];
    for (const opt of options) {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = opt;
      input.appendChild(o);
    }
  } else {
    input = document.createElement("input");
    input.type = "number";
    if (rule) {
      input.min = rule.min;
      input.max = rule.max;
      input.step = rule.integer ? "1" : "any";
      input.title = `Must be between ${rule.min} and ${rule.max}` +
        (rule.integer ? " (whole number)" : "");
    } else {
      input.step = "any";
    }
  }

  input.id = `f-${col}`;
  input.name = col;
  input.required = true;

  // Clear field-level error styling on edit
  input.addEventListener("input", () => {
    input.classList.remove("invalid");
    const err = wrap.querySelector(".field-error");
    if (err) err.remove();
  });

  wrap.appendChild(input);
  return wrap;
}

async function loadSchema() {
  try {
    const res = await fetch(`${API_BASE}/schema`);
    if (!res.ok) throw new Error(`Schema HTTP ${res.status}`);
    SCHEMA = await res.json();

    fieldsEl.innerHTML = "";
    for (const col of SCHEMA.feature_columns) {
      fieldsEl.appendChild(buildField(col));
    }
    fillSample();
  } catch (e) {
    fieldsEl.innerHTML = `<p class="loading">Failed to load schema: ${e.message}</p>`;
  }
}

function fillSample() {
  if (!SCHEMA) return;
  for (const col of SCHEMA.feature_columns) {
    const el = document.getElementById(`f-${col}`);
    if (el && SAMPLE[col] !== undefined) {
      el.value = SAMPLE[col];
    }
  }
  clearErrors();
}

function readForm() {
  const payload = {};
  for (const col of SCHEMA.feature_columns) {
    const el = document.getElementById(`f-${col}`);
    let value = el.value;
    if (!SCHEMA.categorical_cols.includes(col)) {
      value = value === "" ? null : Number(value);
    }
    payload[col] = value;
  }
  return payload;
}

// ---------- Validation helpers ----------
function clearErrors() {
  document.querySelectorAll(".field input.invalid, .field select.invalid")
    .forEach((el) => el.classList.remove("invalid"));
  document.querySelectorAll(".field-error").forEach((el) => el.remove());
  errorEl.classList.add("hidden");
  errorEl.innerHTML = "";
}

function markFieldError(col, message) {
  const el = document.getElementById(`f-${col}`);
  if (!el) return;
  el.classList.add("invalid");
  const wrap = el.closest(".field");
  if (wrap && !wrap.querySelector(".field-error")) {
    const small = document.createElement("small");
    small.className = "field-error";
    small.textContent = message;
    wrap.appendChild(small);
  }
}

function showErrors(messages, summary) {
  resultEl.classList.add("hidden");
  errorEl.classList.remove("hidden");
  const heading = summary || "Please fix the following:";
  errorEl.innerHTML =
    `<strong>${heading}</strong>` +
    `<ul>${messages.map((m) => `<li>${m}</li>`).join("")}</ul>`;
}

function validateClientSide(payload) {
  const errs = [];
  const rules = SCHEMA.validation_rules || {};

  for (const col of SCHEMA.feature_columns) {
    const isCat = SCHEMA.categorical_cols.includes(col);
    const value = payload[col];

    if (isCat) {
      const allowed = SCHEMA.categorical_options[col] || [];
      if (!value || !allowed.includes(String(value))) {
        const msg = `${LABELS[col] || col}: pick one of ${allowed.join(", ")}.`;
        errs.push(msg);
        markFieldError(col, "Required");
      }
      continue;
    }

    if (value === null || value === undefined || Number.isNaN(value)) {
      errs.push(`${LABELS[col] || col}: required.`);
      markFieldError(col, "Required");
      continue;
    }

    const rule = rules[col];
    if (rule) {
      if (value < rule.min || value > rule.max) {
        const msg =
          `${LABELS[col] || col}: must be between ${rule.min} and ${rule.max}.`;
        errs.push(msg);
        markFieldError(col, `Must be ${rule.min}–${rule.max}`);
        continue;
      }
      if (rule.integer && !Number.isInteger(value)) {
        const msg = `${LABELS[col] || col}: must be a whole number.`;
        errs.push(msg);
        markFieldError(col, "Whole number");
        continue;
      }
    }
  }

  return errs;
}

function showResult(data) {
  errorEl.classList.add("hidden");
  resultEl.classList.remove("hidden");

  const isPlaced = data.prediction === "Placed";
  predictionEl.textContent = data.prediction;
  predictionEl.className = "value " + (isPlaced ? "placed" : "not-placed");

  probabilityEl.textContent = `${(data.probability * 100).toFixed(1)}%`;

  barsEl.innerHTML = "";
  const probs = data.class_probabilities || {};
  for (const [cls, p] of Object.entries(probs)) {
    const row = document.createElement("div");
    row.className = "bar";

    const name = document.createElement("span");
    name.textContent = cls;
    name.style.width = "90px";
    row.appendChild(name);

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${(p * 100).toFixed(1)}%`;
    track.appendChild(fill);
    row.appendChild(track);

    const pct = document.createElement("span");
    pct.textContent = `${(p * 100).toFixed(1)}%`;
    pct.style.width = "60px";
    pct.style.textAlign = "right";
    row.appendChild(pct);

    barsEl.appendChild(row);
  }
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!SCHEMA) return;

  clearErrors();
  const payload = readForm();

  // Client-side validation first
  const clientErrs = validateClientSide(payload);
  if (clientErrs.length) {
    showErrors(clientErrs);
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      // Backend may return {error, details: [...]}
      const details = Array.isArray(data.details) ? data.details : null;
      if (details && details.length) {
        showErrors(details, data.error || "Server rejected the input:");
      } else {
        showErrors([data.error || `HTTP ${res.status}`]);
      }
    } else {
      showResult(data);
    }
  } catch (e) {
    showErrors([`Request failed: ${e.message}`]);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict";
  }
});

sampleBtn.addEventListener("click", fillSample);
formEl.addEventListener("reset", () => {
  resultEl.classList.add("hidden");
  clearErrors();
});

// ---------- Tab switching ----------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    document
      .querySelectorAll(".tab")
      .forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `tab-${target}`);
    });
  });
});

loadSchema();
