# AI-Powered Health Assistant App

A comprehensive health and wellness application with AI-powered insights, health data processing from multiple wearable devices, and intelligent calendar optimization.

## 🌟 Features

- **Multi-Device Health Data Processing**: Upload and analyze data from Samsung Health, Apple Health, Fitbit, Garmin, and more
- **AI-Powered Health Analysis**: Google Gemini AI provides personalized health insights and recommendations
- **Smart Calendar Integration**: Google Calendar OAuth integration with AI-powered schedule optimization
- **Health Pattern Recognition**: Machine learning analysis to identify health trends and correlations
- **Intelligent Reminders**: Context-aware health reminders based on your schedule and health data
- **Comprehensive Dashboard**: Real-time health scoring and progress tracking

## 🚀 Deployment on Render.com

### Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Account**: Your code needs to be in a GitHub repository
3. **Google Gemini API Key**: Get from [Google AI Studio](https://ai.google.dev/)
4. **Google Calendar OAuth Credentials** (optional): From [Google Cloud Console](https://console.cloud.google.com)

### Deployment Steps

#### 1. Push to GitHub

```bash
# Navigate to project directory
cd "c:\Users\Aashutosh\OneDrive\Desktop\Coding\Projects\AI-powered health assistant app 2.0\level3"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Ready for Render deployment"

# Add your GitHub repository
git remote add origin YOUR_GITHUB_REPO_URL

# Push to GitHub
git push -u origin main
```

#### 2. Create New Web Service on Render

1. Go to [render.com/dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select your repository from the list
5. Render will auto-detect the `render.yaml` configuration

#### 3. Configure Environment Variables

In the Render dashboard, add these environment variables:

**Required:**
- `GEMINI_API_KEY`: Your Google Gemini API key

**Optional (for Google Calendar features):**
- `GOOGLE_CALENDAR_CLIENT_ID`: Your OAuth 2.0 Client ID
- `GOOGLE_CALENDAR_CLIENT_SECRET`: Your OAuth 2.0 Client Secret

**Auto-configured by render.yaml:**
- `SECRET_KEY`: Auto-generated
- `DATABASE_URL`: Auto-linked to PostgreSQL
- `FLASK_ENV`: Set to production
- `PYTHON_VERSION`: Set to 3.11.0

#### 4. Deploy!

Click **"Create Web Service"** - Render will:
- Install dependencies (~5-10 minutes)
- Download spaCy language model
- Run database migrations
- Start your application

#### 5. Configure Google OAuth (if using Calendar features)

After deployment, your app URL will be: `https://ai-health-assistant.onrender.com`

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **APIs & Services** → **Credentials**
3. Edit your OAuth 2.0 Client ID
4. Add to **Authorized redirect URIs**:
   ```
   https://ai-health-assistant.onrender.com/google_calendar/callback
   ```

## 📊 Free Tier Limits

- **Uptime**: 750 hours/month (auto-spins down after 15 min inactivity)
- **RAM**: 512MB
- **Database**: Free PostgreSQL with backups
- **Storage**: 1GB persistent disk for uploads
- **Build Time**: No limit

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run migrations
flask db upgrade

# Start development server
python run.py
```

## 🌐 Environment Variables

Create a `.env` file for local development:

```env
SECRET_KEY=your-secret-key-here
GEMINI_API_KEY=your-gemini-api-key
GOOGLE_CALENDAR_CLIENT_ID=your-client-id
GOOGLE_CALENDAR_CLIENT_SECRET=your-client-secret
FLASK_ENV=development
```

## 📝 Tech Stack

- **Backend**: Flask 2.3.3, SQLAlchemy, PostgreSQL
- **AI/ML**: Google Gemini 2.5 Flash, scikit-learn, spaCy, OR-Tools
- **Authentication**: Flask-Login, Google OAuth2
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render.com, Gunicorn

## 🐛 Troubleshooting

**Build fails?**
- Check Render logs for specific errors
- Ensure all environment variables are set

**Database issues?**
- Verify DATABASE_URL is auto-linked
- Check migration logs in Render console

**OAuth not working?**
- Confirm redirect URI in Google Cloud Console
- Use exact URL from Render (including https://)

**App spinning down?**
- Free tier spins down after 15 min inactivity
- First request after spin-down takes ~30 seconds
- Upgrade to paid plan ($7/month) for always-on

## 📧 Support

For issues or questions about deployment, check Render's [documentation](https://render.com/docs) or logs in your dashboard.

## 📄 License

This project is for educational and testing purposes.
