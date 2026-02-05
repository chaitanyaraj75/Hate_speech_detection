# Hate Speech Detection System

A full-stack web application for detecting and analyzing hate speech in text content using machine learning. Built with React, Flask, and scikit-learn.

## Features

- **Real-time Detection**: Identify hate speech in user input instantly
- **RESTful API**: Flask backend with CORS support for seamless frontend-backend integration
- **Responsive UI**: Mobile-friendly interface with Tailwind CSS and DaisyUI components
- **Clean UX**: Text analysis with result panels, loading states, and error handling
- **Easy Integration**: Simple JSON API endpoint for predictions

## Tech Stack

### Frontend
- **Framework**: React 19 with Vite (lightning-fast build & HMR)
- **Styling**: Tailwind CSS 4 + DaisyUI 5 (utility-first design system)
- **Routing**: React Router v7
- **HTTP Client**: Axios
- **Code Quality**: ESLint for consistent code style
- **Build Tool**: Vite with @vitejs/plugin-react

### Backend
- **Framework**: Flask 3.1.1 with Flask-CORS
- **ML/Data**: scikit-learn + joblib for model inference
- **Security**: Flask-JWT-Extended for token-based auth
- **Runtime**: Python 3.10+

## Project Structure

```
Hate_speech_detection/
├── backend/
│   ├── app.py                          # Flask app with /predict endpoint
│   ├── requirements.txt                # Python dependencies
│   ├── venv/                           # Virtual environment
│   └── resources/
│       └── hate_speech_pipeline.pkl    # ML model (place here)
└── frontend/
    ├── src/
    │   ├── App.jsx                     # Main app + routing
    │   ├── components/
    │   │   ├── NavBar.jsx              # Navigation bar
    │   │   └── Login.jsx               # Login form
    │   ├── pages/
    │   │   └── Hatespeechdetector.jsx  # Main detection UI
    │   ├── main.jsx                    # Entry point
    │   └── index.css                   # Tailwind + DaisyUI config
    ├── package.json
    ├── vite.config.js
    └── eslint.config.js
```

## Prerequisites

- **Python** 3.10 or higher
- **Node.js** 16+ with npm
- **Git** (optional, for version control)

## Installation & Setup

### Backend Setup

1. Navigate to the backend folder:
```powershell
cd backend
```

2. Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Upgrade pip and install dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. **Add the ML Model**: 
   - Place your `hate_speech_pipeline.pkl` file in `backend/resources/`
   - Or update the path in `app.py` to point to your model location

5. Run the server:
```powershell
python app.py
```
   - Server runs on `http://localhost:3001` by default

### Frontend Setup

1. Navigate to the frontend folder (in a new terminal):
```powershell
cd frontend
```

2. Install dependencies:
```powershell
npm install
```

3. Start the development server:
```powershell
npm run dev
```
   - Open the URL shown in terminal (usually `http://localhost:5173`)

## Usage

1. Enter text or a tweet in the input area
2. Click **Predict** to analyze the content
3. View the result: "Hate Speech 🚨" or "Not Hate Speech ✅"
4. Use **Clear** to reset or **Example** to try sample text

## API Endpoints

### POST `/predict`
Analyze text for hate speech.

**Request:**
```json
{
  "tweet": "Your text here"
}
```

**Response (with model):**
```json
{
  "tweet": "Your text here",
  "prediction": 0
}
```
- `0` = Not hate speech
- `1` = Hate speech

**Error Response:**
```json
{
  "error": "Model file not found or invalid prediction"
}
```

### Other Routes
- `GET /` — Health check (returns "Hello world")
- `GET /health` — Server status
- `GET /greet/<name>` — Greeting endpoint
- `GET /add/<a>/<b>` — Addition example
- `POST/GET/PUT /hello` — HTTP method demo

## Build for Production

### Frontend
```bash
npm run build
```
Output: `dist/` folder (ready for deployment)

### Backend
Run with production settings:
```powershell
$env:FLASK_ENV = "production"
python app.py
```

## Troubleshooting

- **ModuleNotFoundError (sklearn, joblib)**: Run `pip install scikit-learn joblib` in the virtual environment
- **Connection refused on port 3001**: Ensure backend is running; check firewall settings
- **Styles not loading**: Clear browser cache and restart Vite dev server
- **Pipeline file not found**: Check the path in `app.py` or place the pickle file at `backend/resources/hate_speech_pipeline.pkl`

## Contributing

1. Ensure your code passes ESLint checks: `npm run lint`
2. Follow the existing code structure and naming conventions
3. Test your changes locally before submitting
4. For production builds, consider adding TypeScript for better type safety

## Future Enhancements

- [ ] Add user authentication with JWT tokens
- [ ] Dashboard with historical detection results
- [ ] Batch processing for multiple texts
- [ ] Model retraining & version management
- [ ] Export results (CSV/PDF)
- [ ] Dark mode toggle in NavBar

## License

Feel free to use and modify this project

## Collaborators

- Chaitanya Raj — Lead Developer
- Ashish Ranjan Singh — ML Engineer
- Himanshu Singh — Frontend Developer
