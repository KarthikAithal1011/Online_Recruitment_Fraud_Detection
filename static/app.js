// App State
let jobsList = [];
let selectedJobId = null;

// Initialize app on DOM Load
document.addEventListener("DOMContentLoaded", () => {
    checkServerStatus();
    // Periodically check status
    setInterval(checkServerStatus, 15000);

    // Hide validation error on input
    const textInput = document.getElementById("custom-text-input");
    if (textInput) {
        textInput.addEventListener("input", () => {
            const errorDiv = document.getElementById("custom-validation-error");
            if (errorDiv) {
                errorDiv.style.display = "none";
            }
        });
    }
});

// Check FastAPI status
async function checkServerStatus() {
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");

    try {
        const response = await fetch("/api/status");
        if (response.ok) {
            const data = await response.json();
            if (data.model_loaded) {
                if (statusDot) statusDot.className = "status-indicator ready";
                if (statusText) statusText.innerText = `Online (${data.device.toUpperCase()})`;
            } else {
                if (statusDot) statusDot.className = "status-indicator error";
                if (statusText) statusText.innerText = "Model loading error";
            }
        } else {
            if (statusDot) statusDot.className = "status-indicator error";
            if (statusText) statusText.innerText = "Server error";
        }
    } catch (error) {
        if (statusDot) statusDot.className = "status-indicator error";
        if (statusText) statusText.innerText = "Offline";
    }
}

// Switch tabs
function switchTab(tabName) {
    // Hide all contents
    const contents = document.querySelectorAll(".tab-content");
    contents.forEach(content => content.classList.remove("active"));

    // Deactivate all buttons
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => item.classList.remove("active"));

    // Activate current tab and button
    document.getElementById(`view-${tabName}`).classList.add("active");
    document.getElementById(`tab-${tabName}`).classList.add("active");
}

// Fetch job listings from API
async function fetchJobs() {
    const listContainer = document.getElementById("job-list-container");
    listContainer.innerHTML = `
        <div class="loading-state">
            <div class="spinner"></div>
            <p>Scraping live listings or generating fallback jobs...</p>
        </div>
    `;

    try {
        const response = await fetch("/api/scrape?limit=5");
        if (!response.ok) throw new Error("Network response was not ok");

        jobsList = await response.json();
        renderJobList();

        // Reset analysis view on new fetch
        const reportContainer = document.getElementById("analysis-viewport-container");
        reportContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-wand-magic-sparkles"></i>
                <h4>No Active Analysis</h4>
                <p>Select a job listing from the left panel and click "Analyze" to see classification and SHAP word-level details.</p>
            </div>
        `;
        selectedJobId = null;
    } catch (error) {
        listContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-triangle-exclamation text-danger" style="color: var(--accent-red)"></i>
                <h4>Failed to Fetch Jobs</h4>
                <p>Error: ${error.message}. Please check if the backend is running and try again.</p>
            </div>
        `;
    }
}

// Render the list of job cards
function renderJobList() {
    const listContainer = document.getElementById("job-list-container");
    if (jobsList.length === 0) {
        listContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-magnifying-glass"></i>
                <h4>No Jobs Found</h4>
                <p>The scraper returned 0 items. Try fetching again.</p>
            </div>
        `;
        return;
    }

    listContainer.innerHTML = jobsList.map((job, idx) => `
        <div class="job-card ${selectedJobId === idx ? 'selected' : ''}" onclick="selectJob(${idx})" id="job-card-${idx}">
            <div class="job-card-header">
                <div class="job-title">${escapeHtml(job.title)}</div>
                <div class="job-source">${job.is_mock ? 'Demo Mock' : 'Remote.co'}</div>
            </div>
            <div class="job-company">
                <i class="fa-regular fa-building"></i> 
                <span>${escapeHtml(job.company_profile ? job.company_profile.split('.')[0] : 'Unknown Company')}</span>
            </div>
            <div class="job-card-footer">
                <span class="job-meta-desc">${escapeHtml(job.description ? job.description.substring(0, 50) + '...' : '')}</span>
                <button class="btn-card-analyze" onclick="event.stopPropagation(); runAnalysis(${idx})">
                    <i class="fa-solid fa-shield-halved"></i> Analyze
                </button>
            </div>
        </div>
    `).join("");
}

// Select job card
function selectJob(idx) {
    selectedJobId = idx;

    // Update card selection states
    const cards = document.querySelectorAll(".job-card");
    cards.forEach(card => card.classList.remove("selected"));
    document.getElementById(`job-card-${idx}`).classList.add("selected");

    // Set details page in the analysis report viewport
    const job = jobsList[idx];
    const reportContainer = document.getElementById("analysis-viewport-container");

    const company = job.company_profile ? job.company_profile.split('.')[0] : "Company Info Unavailable";

    reportContainer.innerHTML = `
        <div class="report-header">
            <div class="report-job-details">
                <h4>${escapeHtml(job.title)}</h4>
                <p><i class="fa-regular fa-building"></i> ${escapeHtml(company)}</p>
                <p><i class="fa-solid fa-link"></i> <a href="${job.url}" target="_blank" style="color: var(--accent-cyan)">View Original Job Link</a></p>
            </div>
        </div>
        
        <div class="form-group" style="margin-bottom: 24px;">
            <label>Concatenated Job Text Preview</label>
            <div class="email-draft-box" style="height: 250px; white-space: normal; line-height: 1.5; text-align: justify;">
                <strong>Title:</strong> ${escapeHtml(job.title)}<br><br>
                <strong>Company Profile:</strong> ${escapeHtml(job.company_profile || 'None')}<br><br>
                <strong>Description:</strong> ${escapeHtml(job.description || 'None')}<br><br>
                <strong>Requirements:</strong> ${escapeHtml(job.requirements || 'None')}<br><br>
                <strong>Benefits:</strong> ${escapeHtml(job.benefits || 'None')}
            </div>
        </div>
        
        <button class="btn btn-primary" style="width: 100%; justify-content: center;" onclick="runAnalysis(${idx})">
            <i class="fa-solid fa-brain"></i> Trigger RoBERTa & SHAP Model
        </button>
    `;
}

// Run prediction analysis
async function runAnalysis(idx) {
    selectedJobId = idx;
    renderJobList(); // Ensure card select state updates

    const job = jobsList[idx];
    const reportContainer = document.getElementById("analysis-viewport-container");

    // Render loading state inside analysis viewport
    reportContainer.innerHTML = `
        <div class="loading-state">
            <div class="spinner" style="width: 48px; height: 48px; border-width: 4px;"></div>
            <h4>Running Transformers Model...</h4>
            <p>Extracting RoBERTa semantic embeddings, running classifier inference, and generating local SHAP feature explanations (may take 10-15s)...</p>
        </div>
    `;

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                title: job.title,
                company_profile: job.company_profile,
                description: job.description,
                requirements: job.requirements,
                benefits: job.benefits,
                url: job.url
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            const errMsg = errData.detail || "Model inference server error";
            throw new Error(errMsg);
        }

        const result = await response.json();
        renderAnalysisReport(job, result, "analysis-viewport-container");
    } catch (error) {
        reportContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-circle-exclamation text-danger" style="color: var(--accent-red)"></i>
                <h4>Inference Failed</h4>
                <p>Error Details: ${error.message}. Please check console logs.</p>
            </div>
        `;
    }
}

// Helper function to validate job text inputs
function validateJobText(text) {
    const cleaned = text.trim().replace(/\s+/g, ' ');
    if (!cleaned) {
        return { isValid: false, message: "Input text cannot be empty." };
    }

    // Markdown code blocks
    if (text.includes("```")) {
        return { isValid: false, message: "Input contains code blocks (```). Code files or scripts are not allowed." };
    }

    // Regex code patterns
    const codePatterns = [
        /\bdef\s+\w+\s*\(/i,
        /\bclass\s+\w+\s*[:{]/i,
        /\bimport\s+[\w\s,]+/i,
        /\bfrom\s+\w+\s+import\b/i,
        /\bconst\s+\w+\s*=/i,
        /\blet\s+\w+\s*=/i,
        /\bvar\s+\w+\s*=/i,
        /\bfunction\s+\w*\s*\(/i,
        /#include\s*[<"]/i,
        /\bpublic\s+(static\s+)?(class|void|int|double|String)\b/i,
        /console\.log\s*\(/i,
        /print\s*\([^)]*\)/i,
        /<\?php/i,
        /<\/?[a-z1-6]+(\s+[^>]+)*>/i,
        /\bselect\s+.*\s+from\b/i,
        /\{[\s\S]*\}/
    ];

    for (const pattern of codePatterns) {
        if (pattern.test(text)) {
            return { isValid: false, message: "Input appears to contain programming code or developer markup. Please paste a standard job description." };
        }
    }

    // High density of programming symbols
    const specialChars = ['{', '}', '[', ']', ';', '<', '>', '=', '(', ')', '_', '$'];
    let specCount = 0;
    for (let i = 0; i < text.length; i++) {
        if (specialChars.includes(text[i])) {
            specCount++;
        }
    }
    if (text.length > 0 && (specCount / text.length) > 0.08) {
        return { isValid: false, message: "Input contains a high density of programming symbols. Please input a regular text job description." };
    }

    const words = cleaned.split(' ');
    if (words.length < 15) {
        return { isValid: false, message: `Input is too short (found ${words.length} words, minimum 15 required).` };
    }
    if (cleaned.length < 80) {
        return { isValid: false, message: `Input is too short (found ${cleaned.length} characters, minimum 80 required).` };
    }

    const jobKeywords = [
        'job', 'position', 'role', 'work', 'experience', 'requirement', 'requirements',
        'responsibility', 'responsibilities', 'duty', 'duties', 'team', 'skill', 'skills',
        'candidate', 'apply', 'company', 'benefit', 'benefits', 'salary', 'qualification',
        'qualifications', 'employment', 'hire', 'hiring', 'recruitment', 'recruiting'
    ];

    let subMatches = 0;
    const lowerText = cleaned.toLowerCase();
    for (const kw of jobKeywords) {
        if (lowerText.includes(kw)) {
            subMatches++;
        }
    }

    if (subMatches < 2) {
        return {
            isValid: false,
            message: "The input does not appear to be a job posting. Please include job-related details such as role responsibilities, requirements, skills, or company info."
        };
    }

    return { isValid: true, message: "" };
}

// Run custom text analysis
async function analyzeCustom(event) {
    event.preventDefault();
    const textInput = document.getElementById("custom-text-input").value;
    const errorDiv = document.getElementById("custom-validation-error");
    const errorSpan = document.getElementById("custom-validation-error-text");

    // Perform client-side validation
    const validation = validateJobText(textInput);
    if (!validation.isValid) {
        if (errorDiv && errorSpan) {
            errorSpan.innerText = validation.message;
            errorDiv.style.display = "flex";
        }
        return;
    } else {
        if (errorDiv) {
            errorDiv.style.display = "none";
        }
    }

    const btn = document.getElementById("btn-analyze-custom");
    const resultPanel = document.getElementById("custom-result-panel");
    const viewport = document.getElementById("custom-analysis-viewport");

    resultPanel.style.display = "block";
    btn.disabled = true;
    viewport.innerHTML = `
        <div class="loading-state">
            <div class="spinner"></div>
            <h4>Running Inference Pipeline...</h4>
            <p>Analyzing custom layout and generating SHAP values (may take 10-15s)...</p>
        </div>
    `;

    // Scroll results into view
    resultPanel.scrollIntoView({ behavior: 'smooth' });

    try {
        const response = await fetch("/api/analyze-custom", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: textInput })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            const errMsg = errData.detail || "Custom text inference failed";
            throw new Error(errMsg);
        }

        const result = await response.json();
        const dummyJob = {
            title: "Custom Text Input Analysis",
            company_profile: "User Input Source",
            description: textInput,
            url: "N/A"
        };

        renderAnalysisReport(dummyJob, result, "custom-analysis-viewport");
    } catch (error) {
        viewport.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-circle-exclamation text-danger" style="color: var(--accent-red)"></i>
                <h4>Analysis Failed</h4>
                <p>${error.message}</p>
            </div>
        `;
    } finally {
        btn.disabled = false;
    }
}

// Helper to render output metrics into a container
function renderAnalysisReport(job, result, targetContainerId) {
    const container = document.getElementById(targetContainerId);
    const riskPercentage = Math.round(result.fraud_probability * 100);
    const strokeOffset = 251.2 - (251.2 * riskPercentage) / 100;

    const riskLevel = result.risk_level || (result.is_fraudulent ? "high" : "low");
    const isFraud = riskLevel === "high";
    let badgeClass = "level-safe";
    let badgeText = "Low Risk (Safe)";
    let descriptionText = "The model is confident this listing matches normal recruitment patterns.";
    let strokeColor = "var(--accent-green)";

    if (riskLevel === "high") {
        badgeClass = "level-fraud";
        badgeText = "High Risk (Scam Alert)";
        descriptionText = "Warning: The semantic details match characteristics of recruitment job scams (e.g. Telegram interviews, check purchases, high cash returns).";
        strokeColor = "var(--accent-red)";
    } else if (riskLevel === "medium") {
        badgeClass = "level-warning";
        badgeText = "Medium Risk (Caution)";
        descriptionText = "Caution: The listing has some characteristics of suspicious postings. Verify the sender's identity and company domain before applying.";
        strokeColor = "#ff9800";
    }
    const company = job.company_profile ? job.company_profile.split('.')[0] : "Company Profile Unavailable";

    // Render SHAP tags
    let shapHtml = "";
    if (result.shap_indicators && result.shap_indicators.length > 0) {
        shapHtml = result.shap_indicators.map(ind => `
            <div class="word-badge" style="background: rgba(255, 56, 96, ${Math.min(0.25, ind.score * 1.5)}); border-color: rgba(255, 56, 96, ${Math.min(0.5, 0.1 + ind.score * 2)})">
                "${escapeHtml(ind.word)}"
                <span>+${ind.score.toFixed(3)}</span>
            </div>
        `).join("");
    } else {
        shapHtml = `<p class="no-indicators">No significant fraudulent keywords detected by SHAP values.</p>`;
    }

    // Render alert box
    let emailHtml = "";
    if (isFraud && result.email_draft) {
        emailHtml = `
            <div class="email-section">
                <h5><i class="fa-solid fa-envelope-open-text"></i> Candidate Warning Dispatch (Draft)</h5>
                <div class="email-box-container">
                    <pre class="email-draft-box" id="email-draft-text">${escapeHtml(result.email_draft)}</pre>
                    <button class="btn-copy" onclick="copyEmailDraft()"><i class="fa-regular fa-copy"></i> Copy Draft</button>
                </div>
            </div>
        `;
    }

    container.innerHTML = `
        <div class="report-header">
            <div class="report-job-details">
                <h4>${escapeHtml(job.title)}</h4>
                <p><i class="fa-regular fa-building"></i> ${escapeHtml(company)}</p>
            </div>
        </div>
        
        <div class="gauge-section">
            <div class="radial-gauge">
                <svg>
                    <circle class="radial-bg" cx="50" cy="50" r="40"></circle>
                    <circle class="radial-progress" cx="50" cy="50" r="40" 
                            style="stroke-dashoffset: ${strokeOffset}; stroke: ${strokeColor};"></circle>
                </svg>
                <div class="radial-value" style="color: ${strokeColor}">${riskPercentage}%</div>
            </div>
            
            <div class="risk-summary">
                <span class="risk-level-badge ${badgeClass}">${badgeText}</span>
                <p class="risk-description">${descriptionText}</p>
            </div>
        </div>
        
        <div class="shap-section">
            <h5><i class="fa-solid fa-key"></i> Key Scam Indicators (SHAP Token Attribution)</h5>
            <div class="indicators-list">
                ${shapHtml}
            </div>
        </div>
        
        ${emailHtml}
    `;
}

// Copy email draft helper
function copyEmailDraft() {
    const draftText = document.getElementById("email-draft-text").innerText;
    navigator.clipboard.writeText(draftText).then(() => {
        const copyBtn = document.querySelector(".btn-copy");
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
        setTimeout(() => {
            copyBtn.innerHTML = originalText;
        }, 2000);
    }).catch(err => {
        alert("Failed to copy text: " + err);
    });
}

// Helper to escape HTML tags
function escapeHtml(unsafe) {
    if (!unsafe) return "";
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
