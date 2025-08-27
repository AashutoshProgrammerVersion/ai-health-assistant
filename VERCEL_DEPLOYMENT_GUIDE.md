# AI-Powered Health Assistant - Vercel Deployment Guide

## Complete Step-by-Step Guide to Deploy on Vercel (Web Interface)

### 🚀 Pre-Deployment Checklist

Your Flask application is now properly configured for Vercel deployment with all necessary files:

- ✅ `wsgi.py` - WSGI entry point
- ✅ `api/index.py` - Vercel API endpoint
- ✅ `vercel.json` - Vercel configuration
- ✅ `requirements.txt` - Production dependencies
- ✅ `config_production.py` - Production settings
- ✅ `runtime.txt` - Python version specification
- ✅ `.vercelignore` - Files to exclude from deployment

### 📁 Your App Features (All Will Work on Vercel)

1. **User Authentication System**
   - User registration and login
   - Secure password hashing
   - Session management

2. **Health Data Management**
   - Manual health data entry
   - File upload from wearable devices (CSV, JSON)
   - Comprehensive health metrics tracking

3. **AI-Powered Features**
   - Google Gemini AI health recommendations
   - Health pattern analysis
   - Personalized advice generation

4. **Calendar Integration**
   - Google Calendar OAuth integration
   - Schedule optimization
   - Smart reminders

5. **Dashboard & Analytics**
   - Health score calculations
   - Visual health metrics
   - Progress tracking

### 🌐 Step-by-Step Vercel Deployment (Web Interface)

#### Step 1: Prepare Your Repository

1. **Upload to GitHub:**
   - Create a new repository on GitHub
   - Upload your `level3` folder contents to the repository
   - Make sure all files are committed

#### Step 2: Connect to Vercel

1. **Visit Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Sign up/log in with your GitHub account

2. **Import Project:**
   - Click "New Project"
   - Select "Import Git Repository"
   - Choose your health assistant repository
   - Click "Import"

#### Step 3: Configure Project Settings

1. **Project Configuration:**
   - **Framework Preset:** Select "Other"
   - **Root Directory:** Keep as default (if you uploaded level3 contents to root)
   - **Build Command:** Leave empty
   - **Output Directory:** Leave empty
   - **Install Command:** `pip install -r requirements.txt`

#### Step 4: Set Environment Variables

In the Vercel project settings, add these environment variables:

**Required Variables:**
```
SECRET_KEY=your-super-secret-production-key-make-it-long-and-random
GEMINI_API_KEY=your-actual-gemini-api-key-from-google
GOOGLE_CALENDAR_CLIENT_ID=your-google-calendar-client-id
GOOGLE_CALENDAR_CLIENT_SECRET=your-google-calendar-client-secret
FLASK_ENV=production
```

**Optional Variables:**
```
DATABASE_URL=your-postgresql-url-if-using-postgres
DEBUG=false
```

#### Step 5: Deploy

1. Click "Deploy"
2. Wait for deployment to complete (usually 2-3 minutes)
3. Your app will be available at `your-project-name.vercel.app`

### 🔑 Setting Up API Keys

#### Google Gemini AI API

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Copy the key and add it as `GEMINI_API_KEY` in Vercel

#### Google Calendar API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Calendar API
4. Create OAuth 2.0 credentials
5. Add your Vercel domain to authorized origins:
   - `https://your-project-name.vercel.app`
6. Add redirect URIs:
   - `https://your-project-name.vercel.app/google_calendar/callback`

### 🗄️ Database Configuration

#### Option 1: SQLite (Simplest)
- No additional setup needed
- Uses file-based database
- Good for development and small-scale use

#### Option 2: PostgreSQL (Recommended for Production)
- Sign up for a free PostgreSQL database at:
  - [Supabase](https://supabase.com/) (recommended)
  - [Railway](https://railway.app/)
  - [PlanetScale](https://planetscale.com/)
- Get the connection URL
- Add it as `DATABASE_URL` environment variable in Vercel

### 🧪 Testing Your Deployment

After deployment, test these features:

1. **Basic Functionality:**
   - Visit your site
   - Register a new account
   - Log in successfully

2. **Health Data:**
   - Navigate to "Log Data"
   - Enter some health metrics
   - Check the dashboard updates

3. **AI Features:**
   - Upload a health data file (CSV/JSON)
   - Check if AI recommendations appear
   - Verify health scoring works

4. **Calendar Integration:**
   - Go to Settings/Preferences
   - Try connecting Google Calendar
   - Test calendar sync

### 🔧 Troubleshooting Common Issues

#### Deployment Fails
- Check build logs in Vercel dashboard
- Ensure all required environment variables are set
- Verify requirements.txt syntax

#### Database Errors
- Check if DATABASE_URL is properly formatted
- Ensure database credentials are correct
- Try using SQLite first for testing

#### API Key Issues
- Verify Gemini API key is valid
- Check Google Calendar OAuth setup
- Ensure redirect URIs match your domain

#### Import Errors
- Check if all dependencies are in requirements.txt
- Verify Python version compatibility
- Look for missing import statements

### 📊 Performance Optimization

Your app is configured for optimal Vercel performance:

- **Function timeout:** 30 seconds
- **Max Lambda size:** 15MB
- **Static file caching:** 1 year
- **Optimized dependencies:** Production-ready packages

### 🔒 Security Features

- CSRF protection enabled
- Secure session management
- Environment-based configuration
- SQL injection protection via SQLAlchemy ORM

### 📈 Monitoring and Maintenance

1. **Monitor in Vercel:**
   - Check function logs
   - Monitor performance metrics
   - Set up error alerts

2. **Regular Updates:**
   - Keep dependencies updated
   - Monitor API usage limits
   - Backup database regularly

### 🎉 Success!

Once deployed, your AI-powered health assistant will have:

- ✅ Full user authentication
- ✅ Health data tracking and analysis
- ✅ AI-powered recommendations
- ✅ Google Calendar integration
- ✅ File upload processing
- ✅ Responsive web interface
- ✅ Production-ready configuration

Your app will be accessible at: `https://your-project-name.vercel.app`

### 📞 Support

If you encounter issues:
1. Check Vercel function logs
2. Verify environment variables
3. Test API keys separately
4. Review the error messages in detail

All features of your health assistant app are now properly configured to work on Vercel!
