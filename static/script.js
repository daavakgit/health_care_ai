/**
 * script.js — MediPredict AI Frontend Logic
 * Handles form validation, API calls, animations, and demo data.
 */

'use strict';

/* ── FORM SUBMISSION ────────────────────────────────────────── */
async function submitForm() {
  const btn      = document.getElementById('submit-btn');
  const btnText  = document.getElementById('btn-text');
  const spinner  = document.getElementById('btn-spinner');
  const arrow    = document.getElementById('btn-arrow');
  const alertBox = document.getElementById('alert-box');

  // Gather & validate all fields
  const data = gatherFormData();
  if (!data) return;   // validation failed, error already shown

  // UI → loading state
  btn.disabled = true;
  btnText.textContent = 'Analysing …';
  spinner.classList.remove('hidden');
  arrow.classList.add('hidden');
  alertBox.classList.add('hidden');

  try {
    const resp = await fetch('/submit_data', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(data),
    });

    const result = await resp.json();

    if (result.success) {
      btnText.textContent = '✓ Done! Redirecting …';
      setTimeout(() => window.location.href = result.redirect, 500);
    } else {
      showError(alertBox, result.error || 'Prediction failed. Please try again.');
      resetBtn(btn, btnText, spinner, arrow);
    }

  } catch (err) {
    showError(alertBox, 'Network error: could not reach the server. Is Flask running?');
    resetBtn(btn, btnText, spinner, arrow);
  }
}


/* ── FORM DATA COLLECTION & VALIDATION ─────────────────────── */
function gatherFormData() {
  const fields = {
    // Non-required personal fields
    name:   document.getElementById('name')?.value.trim() || 'Patient',
    gender: document.getElementById('gender')?.value     || '—',

    // Required numeric fields
    age:               getFloat('age'),
    bmi:               getFloat('bmi'),
    blood_pressure:    getFloat('blood_pressure'),
    glucose:           getFloat('glucose'),
    cholesterol:       getFloat('cholesterol'),
    hba1c:             getFloat('hba1c'),
    creatinine:        getFloat('creatinine'),
    physical_activity: getFloat('physical_activity'),

    // Required select fields
    smoking:        getSelect('smoking'),
    family_history: getSelect('family_history'),
  };

  // Check for missing required numeric fields
  const requiredNumerics = [
    'age','bmi','blood_pressure','glucose','cholesterol',
    'hba1c','creatinine','physical_activity'
  ];
  for (const key of requiredNumerics) {
    if (fields[key] === null) {
      flashField(key);
      showError(document.getElementById('alert-box'),
        `Please fill in the "${labelFor(key)}" field.`);
      return null;
    }
  }

  // Check select fields
  if (fields.smoking === null) {
    flashField('smoking');
    showError(document.getElementById('alert-box'), 'Please select your smoking status.');
    return null;
  }
  if (fields.family_history === null) {
    flashField('family_history');
    showError(document.getElementById('alert-box'), 'Please select family history.');
    return null;
  }

  return fields;
}

function getFloat(id) {
  const el = document.getElementById(id);
  if (!el || el.value.trim() === '') return null;
  const v = parseFloat(el.value);
  return isNaN(v) ? null : v;
}

function getSelect(id) {
  const el = document.getElementById(id);
  if (!el || el.value === '') return null;
  return parseFloat(el.value);
}

function flashField(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.borderColor = '#ef4444';
  el.style.boxShadow   = '0 0 0 3px rgba(239,68,68,0.25)';
  el.addEventListener('input', () => {
    el.style.borderColor = '';
    el.style.boxShadow   = '';
  }, { once: true });
  el.focus();
}

function labelFor(fieldId) {
  const map = {
    age: 'Age', bmi: 'BMI', blood_pressure: 'Blood Pressure',
    glucose: 'Glucose', cholesterol: 'Cholesterol',
    hba1c: 'HbA1c', creatinine: 'Creatinine',
    physical_activity: 'Physical Activity'
  };
  return map[fieldId] || fieldId;
}

function showError(el, msg) {
  if (!el) return;
  el.textContent = '⚠️  ' + msg;
  el.classList.remove('hidden');
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetBtn(btn, btnText, spinner, arrow) {
  btn.disabled = false;
  btnText.textContent = 'Analyse My Health Risk';
  spinner.classList.add('hidden');
  arrow.classList.remove('hidden');
}


/* ── DEMO DATA FILLER ───────────────────────────────────────── */
const SAMPLE_DATA = {
  high: {
    name: 'John Sample', gender: 'Male',
    age: 62, bmi: 34.2, blood_pressure: 158,
    glucose: 245, cholesterol: 298, hba1c: 9.1, creatinine: 3.4,
    smoking: '1', physical_activity: 1, family_history: '1',
  },
  low: {
    name: 'Emma Sample', gender: 'Female',
    age: 28, bmi: 22.5, blood_pressure: 112,
    glucose: 84, cholesterol: 162, hba1c: 5.1, creatinine: 0.8,
    smoking: '0', physical_activity: 5, family_history: '0',
  },
};

function fillSampleData(profile) {
  const d = SAMPLE_DATA[profile];
  if (!d) return;

  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.value = val;
  };

  set('name', d.name); set('gender', d.gender);
  set('age', d.age); set('bmi', d.bmi);
  set('blood_pressure', d.blood_pressure);
  set('glucose', d.glucose); set('cholesterol', d.cholesterol);
  set('hba1c', d.hba1c); set('creatinine', d.creatinine);
  set('smoking', d.smoking);
  set('physical_activity', d.physical_activity);
  set('family_history', d.family_history);

  // Brief visual feedback
  const demoBar = document.querySelector('.demo-bar span');
  if (demoBar) {
    const orig = demoBar.textContent;
    demoBar.textContent = `✓ ${profile === 'high' ? 'High-risk' : 'Low-risk'} sample loaded!`;
    setTimeout(() => { demoBar.textContent = orig; }, 2000);
  }
}


/* ── RESULT PAGE ANIMATIONS ─────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  // Animate probability and factor bars on scroll (Intersection Observer)
  if ('IntersectionObserver' in window) {
    const bars = document.querySelectorAll('.proba-bar, .factor-bar');

    // Store the intended widths, then reset to 0 for animation
    bars.forEach(bar => {
      bar.dataset.targetWidth = bar.style.width;
      bar.style.width = '0%';
    });

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const bar = entry.target;
          setTimeout(() => {
            bar.style.width = bar.dataset.targetWidth;
          }, 80);
          observer.unobserve(bar);
        }
      });
    }, { threshold: 0.2 });

    bars.forEach(bar => observer.observe(bar));
  }

  // Smooth scroll on anchor links
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      const target = document.querySelector(a.getAttribute('href'));
      if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
  });

  // Stagger-animate cards on home page
  document.querySelectorAll('.step-card, .disease-card, .stat-card').forEach((card, i) => {
    card.style.animationDelay = `${i * 0.1}s`;
    card.classList.add('fade-in-card');
  });
});
