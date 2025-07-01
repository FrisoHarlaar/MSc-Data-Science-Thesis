# Novelscape: Emotion-Based Book Recommendation Web Application

This Django web application provides an interactive interface for the emotion-based book recommendation system. Users can upload images to receive book recommendations based on the emotional content detected by trained deep learning models.

![Web Application Workflow](../figures/webapp_workflow.png)

## Features

### Functionality
- **Image Upload**: Upload any image to get emotion-based book recommendations
- **Emotion Classification**: Uses trained ResNet model for to classify image
- **Book Recommendations**: Finds books with similar emotional profiles using cosine similarity 
- **Real-time Processing**: Fast inference using PostgreSQL with PGVector for similarity search

## Technology Stack

### Backend
- **Django 5.1**: Web framework
- **Wagtail**: Content Management System
- **PostgreSQL**: Database with PGVector extension for similarity search
- **PyTorch**: For model inference

Checkout the [requirements](requirements.txt)

### Deployment
- **Docker**: Containerized deployment
- **Gunicorn**: WSGI HTTP Server

## Installation & Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 14+ with PGVector extension
- Docker (for containerized deployment)

### Docker Deployment

#### 1. Build and Run with Docker
```bash
# Build the container
docker build -t novelscape .

# Run with environment variables
docker run -d \
  --name novelscape \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  -e SECRET_KEY=your-secret-key \
  -v $(pwd)/media:/app/media \
  novelscape
```

#### 2. Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# Run migrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser
```

## Application Structure

```
website/
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Multi-container setup
├── website/                 # Main Django project
│   ├── settings/            # Environment-specific settings
│   ├── urls.py             # URL routing
│   ├── wsgi.py             # WSGI configuration
│   └── templates/          # Base templates
├── books/                   # Book recommendation app
│   ├── models.py           # Book and emotion models
│   ├── views.py            # Recommendation logic
│   ├── ml_models.py        # ML model integration
│   └── management/         # Data import commands
├── users/                   # User management app
│   ├── models.py           # User profile models
│   ├── views.py            # Authentication views
│   └── templates/          # User templates
├── home/                    # Homepage app
│   ├── models.py           # CMS page models
│   └── templates/          # Homepage templates
└── static/                  # Static assets
    ├── css/                # Stylesheets
    ├── js/                 # JavaScript
    └── images/             # Images and icons
```

## Configuration

### Environment Variables
```bash
# Required
SECRET_KEY=your-django-secret-key
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Optional
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,localhost
EMAIL_URL=smtp://user:pass@smtp.example.com:587

# Model paths
RESNET_MODEL_PATH=../artemis_scripts/models/resnet.pt
```

## Performance Considerations

To host the website a (not very strong) GPU is needed to load the ResNet checkpoint and run the inference on the input images, which is why it is hosted locally right now.