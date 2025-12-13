# Quickstart Guide for AI-Driven Physical AI & Humanoid Robotics Book

## Prerequisites
- Node.js 18+ installed
- Python 3.11+ installed
- Git installed
- Access to Neon Postgres account

## Setting Up the Backend (FastAPI)

1. **Navigate to the backend directory**
   ```bash
   cd backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn pydantic python-jose[cryptography] passlib[bcrypt] psycopg2-binary python-dotenv
   ```

4. **Set up environment variables**
   Create `.env` file with:
   ```
   DATABASE_URL=your_neon_postgres_connection_string
   SECRET_KEY=your_jwt_secret_key
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

5. **Run the backend server**
   ```bash
   uvicorn src.main:app --reload --port 8000
   ```

## Setting Up the Frontend (Docusaurus)

1. **Navigate to the robotics-book directory**
   ```bash
   cd robotics-book
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Run the development server**
   ```bash
   npm start
   ```

## Environment Configuration

### Backend (.env)
```
DATABASE_URL=postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname
SECRET_KEY=generate_a_secure_random_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
DEBUG=true
```

### Frontend (docusaurus.config.js)
```javascript
module.exports = {
  // ... existing config
  customFields: {
    backendUrl: process.env.BACKEND_URL || 'http://localhost:8000',
    authEnabled: true,
  },
};
```

## API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `POST /auth/refresh` - Token refresh
- `GET /auth/me` - Get current user (requires auth)

### Users
- `GET /users/me` - Get current user details
- `PUT /users/me` - Update user profile

### Modules
- `GET /modules/` - List all modules
- `GET /modules/{id}` - Get specific module
- `GET /modules/{id}/lessons` - Get lessons in a module

### Lessons
- `GET /lessons/{id}` - Get lesson content (requires auth if not free)
- `POST /lessons/{id}/progress` - Update lesson progress

## Running Tests

### Backend
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

### Frontend
```bash
# Run tests
npm test

# Run linting
npm run lint
```

## Database Migrations
The application uses Alembic for database migrations:
```bash
# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head
```

## Building for Production

### Backend
```bash
# Build Docker image (optional)
docker build -t robotics-book-backend .
```

### Frontend
```bash
# Build static site
npm run build

# Serve built site
npm run serve
```

## Admin Setup
To create an initial admin user, use the backend CLI or directly insert into the database with role='admin'.