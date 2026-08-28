# Fix Broken Recruitment Links for Mock Jobs

## User Review Required
No breaking changes are introduced. The application will serve mock recruitment pages locally instead of redirecting to the non-existent `mock-job-board.com`.

## Proposed Changes

### Scraping / Mock Feed Component

#### [MODIFY] [live_scraper.py](file:///d:/ORF/live_scraper.py)
Change mock job URLs from `http://mock-job-board.com/jobs/10X` to `/mock-job/10X`.

---

### Backend API Server

#### [MODIFY] [app.py](file:///d:/ORF/app.py)
Add a new route `@app.get("/mock-job/{job_id}", response_class=HTMLResponse)` to render a premium styled job posting page using the attributes of the mock job (e.g. Title, Company, Description, Requirements, Benefits).

## Verification Plan

### Automated Tests
- None.

### Manual Verification
1. Run `python app.py` to start the backend.
2. Open `http://127.0.0.1:8000`.
3. Click "Fetch Jobs" (forces fallback to mock jobs if remote.co fails or if internet connection is restricted).
4. Click on any job card.
5. Click "View Original Job Link".
6. Verify that it opens in a new tab showing a beautifully formatted mock job listing page with a clean theme, instead of a "page not found" error.
