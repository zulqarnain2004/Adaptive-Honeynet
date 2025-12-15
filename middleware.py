"""
Middleware for Adaptive Deception Mesh API
"""

from flask import request, jsonify, g
import time
import hashlib
from functools import wraps
import jwt
from datetime import datetime, timedelta
import os

class RateLimiter:
    """Rate limiting middleware"""
    
    def __init__(self, max_requests=100, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}
    
    def is_allowed(self, client_id):
        """Check if client is allowed to make request"""
        current_time = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < self.window
        ]
        
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(current_time)
            return True
        
        return False

class SecurityMiddleware:
    """Security middleware for API protection"""
    
    def __init__(self, app=None):
        self.app = app
        self.jwt_secret = os.environ.get('JWT_SECRET', 'adaptive-deception-mesh-secret-key')
        
    def init_app(self, app):
        """Initialize middleware with app"""
        self.app = app
        
    def generate_token(self, user_id, expires_in=3600):
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_token(self, token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

class RequestLogger:
    """Request logging middleware"""
    
    def __init__(self, app=None):
        self.app = app
        
    def init_app(self, app):
        """Initialize with app"""
        @app.before_request
        def before_request():
            g.start_time = time.time()
            g.request_id = hashlib.md5(
                f"{request.remote_addr}{time.time()}".encode()
            ).hexdigest()[:8]
            
            # Log request
            app.logger.info(f"Request {g.request_id}: {request.method} {request.path} "
                          f"from {request.remote_addr}")
            
            # Log request body for non-GET requests
            if request.method != 'GET' and request.content_length and request.content_length < 1024:
                try:
                    app.logger.debug(f"Request {g.request_id} body: {request.get_json(silent=True)}")
                except:
                    pass
        
        @app.after_request
        def after_request(response):
            if hasattr(g, 'start_time'):
                duration = time.time() - g.start_time
                response.headers['X-Response-Time'] = f'{duration:.3f}s'
                
                app.logger.info(f"Response {g.request_id}: {response.status_code} "
                              f"in {duration:.3f}s")
            
            # Add security headers
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            
            # CORS headers
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            
            return response

class AuthenticationMiddleware:
    """Authentication middleware"""
    
    def __init__(self, app=None):
        self.app = app
        self.security = SecurityMiddleware()
        
    def init_app(self, app):
        """Initialize with app"""
        self.app = app
        self.security.init_app(app)
        
    def require_auth(self, f):
        """Decorator to require authentication"""
        @wraps(f)
        def decorated(*args, **kwargs):
            token = None
            
            # Get token from header
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
            
            if not token:
                return jsonify({'error': 'Authentication required'}), 401
            
            # Verify token
            payload = self.security.verify_token(token)
            if not payload:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            # Add user info to request context
            g.user_id = payload['user_id']
            g.token_payload = payload
            
            return f(*args, **kwargs)
        return decorated

class ValidationMiddleware:
    """Request validation middleware"""
    
    def validate_network_data(self, data):
        """Validate network data for attack detection"""
        required_fields = ['src_ip', 'dst_ip', 'packet_count']
        
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate IP addresses
        import re
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        
        if not re.match(ip_pattern, data.get('src_ip', '')):
            return False, "Invalid source IP format"
        
        if not re.match(ip_pattern, data.get('dst_ip', '')):
            return False, "Invalid destination IP format"
        
        # Validate numeric fields
        if not isinstance(data.get('packet_count'), (int, float)) or data['packet_count'] < 0:
            return False, "Packet count must be a positive number"
        
        if 'duration' in data and (not isinstance(data['duration'], (int, float)) or data['duration'] < 0):
            return False, "Duration must be a positive number"
        
        return True, "Valid"
    
    def validate_simulation_config(self, data):
        """Validate simulation configuration"""
        allowed_fields = ['duration', 'attack_intensity', 'network_size']
        
        if not isinstance(data, dict):
            return False, "Configuration must be a dictionary"
        
        for key, value in data.items():
            if key not in allowed_fields:
                return False, f"Unknown configuration field: {key}"
            
            if not isinstance(value, (int, float)):
                return False, f"{key} must be a number"
            
            # Range checks
            if key == 'duration' and (value < 1 or value > 3600):
                return False, "Duration must be between 1 and 3600 seconds"
            
            if key == 'attack_intensity' and (value < 0 or value > 1):
                return False, "Attack intensity must be between 0 and 1"
            
            if key == 'network_size' and (value < 10 or value > 100):
                return False, "Network size must be between 10 and 100 nodes"
        
        return True, "Valid"

# Initialize middleware instances
rate_limiter = RateLimiter(max_requests=100, window=60)
request_logger = RequestLogger()
auth_middleware = AuthenticationMiddleware()
validation_middleware = ValidationMiddleware()

def rate_limit(f):
    """Rate limiting decorator"""
    @wraps(f)
    def decorated(*args, **kwargs):
        client_id = request.remote_addr
        
        if not rate_limiter.is_allowed(client_id):
            return jsonify({
                'error': 'Rate limit exceeded',
                'retry_after': rate_limiter.window
            }), 429
        
        return f(*args, **kwargs)
    return decorated

def validate_request(schema='network_data'):
    """Request validation decorator"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            data = request.get_json(silent=True)
            
            if schema == 'network_data':
                is_valid, message = validation_middleware.validate_network_data(data)
            elif schema == 'simulation_config':
                is_valid, message = validation_middleware.validate_simulation_config(data)
            else:
                is_valid, message = True, "No validation specified"
            
            if not is_valid:
                return jsonify({'error': f'Invalid request: {message}'}), 400
            
            return f(*args, **kwargs)
        return decorated
    return decorator