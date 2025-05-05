/**
 * Tailwind CSS Configuration
 * 
 * This script ensures that we're using a locally built version of Tailwind CSS
 * instead of loading it from a CDN, which is not recommended for production.
 */
(function() {
  // Find and remove any CDN Tailwind CSS link tags
  document.addEventListener('DOMContentLoaded', function() {
    // Look for any script or link tags loading Tailwind from CDN
    const links = document.querySelectorAll('link[href*="tailwindcss"]');
    const scripts = document.querySelectorAll('script[src*="tailwindcss"]');
    
    // Remove any CDN links found
    links.forEach(link => {
      console.log('Removing Tailwind CDN link:', link.href);
      link.remove();
    });
    
    // Remove any CDN scripts found
    scripts.forEach(script => {
      console.log('Removing Tailwind CDN script:', script.src);
      script.remove();
    });
    
    // Add the local Tailwind CSS file if needed
    if (!document.querySelector('link[href*="/static/css/tailwind.css"]')) {
      const localLink = document.createElement('link');
      localLink.rel = 'stylesheet';
      localLink.href = '/static/css/tailwind.css';
      document.head.appendChild(localLink);
      console.log('Added local Tailwind CSS file');
    }
  });
})(); 