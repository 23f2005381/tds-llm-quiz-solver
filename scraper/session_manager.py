# =============================================================================
# FILE: src/scrapers/session_manager.py
# =============================================================================
"""
Advanced session management with automatic cookie handling, persistence,
and authentication support for web scraping.
References: [web:292][web:294][web:296][web:306][web:308][web:310]
"""

import requests
from http.cookiejar import CookieJar, MozillaCookieJar
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
import structlog
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class SessionManager:
    """
    Manage HTTP sessions with automatic cookie handling and persistence.
    Maintains authentication state across multiple requests[web:292][web:294].
    """
    
    def __init__(self, session_file: Optional[Path] = None):
        """
        Initialize session manager.
        
        Args:
            session_file: Optional path to save/load session cookies
        """
        self.session = requests.Session()
        self.session_file = session_file
        self._configure_session()
        
        if session_file and session_file.exists():
            self.load_session()
    
    def _configure_session(self):
        """Configure session with default headers and settings[web:292]"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Configure connection pooling and keep-alive
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3,
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def get(self, url: str, **kwargs) -> requests.Response:
        """
        Make GET request with automatic cookie handling[web:294][web:306].
        
        Args:
            url: Target URL
            **kwargs: Additional requests arguments
            
        Returns:
            Response object
        """
        logger.info("session_get", url=url)
        response = self.session.get(url, **kwargs)
        response.raise_for_status()
        return response
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def post(self, url: str, **kwargs) -> requests.Response:
        """
        Make POST request with automatic cookie handling[web:296].
        
        Args:
            url: Target URL
            **kwargs: Additional requests arguments
            
        Returns:
            Response object
        """
        logger.info("session_post", url=url)
        response = self.session.post(url, **kwargs)
        response.raise_for_status()
        return response
    
    def login(
        self,
        login_url: str,
        credentials: Dict[str, str],
        method: str = 'POST'
    ) -> bool:
        """
        Perform login and persist session cookies[web:308][web:310].
        
        Args:
            login_url: Login endpoint URL
            credentials: Dict with username, password, etc.
            method: HTTP method (POST or GET)
            
        Returns:
            True if login successful
        """
        try:
            if method.upper() == 'POST':
                response = self.post(login_url, data=credentials)
            else:
                response = self.get(login_url, params=credentials)
            
            # Check if login was successful (basic heuristic)
            success = response.status_code == 200 and len(self.session.cookies) > 0
            
            if success:
                logger.info("login_successful", url=login_url)
                # Save session if configured
                if self.session_file:
                    self.save_session()
            else:
                logger.warning("login_failed", url=login_url, status=response.status_code)
            
            return success
            
        except Exception as e:
            logger.error("login_error", url=login_url, error=str(e))
            return False
    
    def save_session(self):
        """
        Save session cookies to file for persistence[web:308][web:310].
        Allows resuming sessions across script executions.
        """
        if not self.session_file:
            logger.warning("No session file configured")
            return
        
        try:
            # Save cookies using pickle
            with open(self.session_file, 'wb') as f:
                pickle.dump(self.session.cookies, f)
            
            logger.info("session_saved", filepath=str(self.session_file))
            
        except Exception as e:
            logger.error("session_save_error", error=str(e))
    
    def load_session(self):
        """
        Load previously saved session cookies[web:308][web:310].
        
        Returns:
            True if cookies loaded successfully
        """
        if not self.session_file or not self.session_file.exists():
            return False
        
        try:
            with open(self.session_file, 'rb') as f:
                self.session.cookies.update(pickle.load(f))
            
            logger.info("session_loaded", filepath=str(self.session_file))
            return True
            
        except Exception as e:
            logger.error("session_load_error", error=str(e))
            return False
    
    def get_cookies(self) -> Dict[str, str]:
        """
        Get all cookies as dictionary.
        
        Returns:
            Dict of cookie name-value pairs
        """
        return requests.utils.dict_from_cookiejar(self.session.cookies)
    
    def set_cookie(self, name: str, value: str, domain: str = None):
        """
        Manually set a cookie[web:292][web:296].
        
        Args:
            name: Cookie name
            value: Cookie value
            domain: Optional domain
        """
        cookie = requests.cookies.create_cookie(
            name=name,
            value=value,
            domain=domain
        )
        self.session.cookies.set_cookie(cookie)
        logger.info("cookie_set", name=name, domain=domain)
    
    def clear_cookies(self):
        """Clear all session cookies"""
        self.session.cookies.clear()
        logger.info("cookies_cleared")
    
    def close(self):
        """Close session and cleanup"""
        self.session.close()
        logger.info("session_closed")
