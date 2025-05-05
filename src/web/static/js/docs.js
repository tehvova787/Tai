document.addEventListener('DOMContentLoaded', function() {
    // Handle sidebar navigation
    const sidebarLinks = document.querySelectorAll('.sidebar-link');
    const docsSections = document.querySelectorAll('.docs-section');
    
    sidebarLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            
            // Update active link
            sidebarLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Smooth scroll to section
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                window.scrollTo({
                    top: targetSection.offsetTop - 100,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Handle browser extension downloads
    const browserDownloads = document.querySelectorAll('.browser-download');
    
    browserDownloads.forEach(download => {
        download.addEventListener('click', function(e) {
            e.preventDefault();
            
            const browser = this.querySelector('span').textContent;
            alert(`The ${browser} extension will be available soon!`);
        });
    });
    
    // Handle scroll to highlight current section in sidebar
    window.addEventListener('scroll', function() {
        const scrollPosition = window.scrollY;
        
        docsSections.forEach(section => {
            const sectionTop = section.offsetTop - 120;
            const sectionBottom = sectionTop + section.offsetHeight;
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                const id = section.getAttribute('id');
                
                sidebarLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    });
}); 