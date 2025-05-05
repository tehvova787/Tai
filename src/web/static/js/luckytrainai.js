/**
 * LuckyTrainAI - Advanced Interface Functionality
 * 
 * This script handles the enhanced UI features for LuckyTrainAI, including:
 * - Futuristic loading screen
 * - Voice input/output via Mozilla Common Voice
 * - Advanced animations and transitions
 * - Holographic UI effects
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the loading screen
    initializeLoadingScreen();
    
    // Initialize voice functionality when page is fully loaded
    window.addEventListener('load', function() {
        initializeVoiceFeatures();
    });
    
    // Initialize dynamic color effects
    initializeColorEffects();
    
    // Fix mobile viewport height
    setMobileViewportHeight();
    
    // Initialize mobile menu
    initializeMobileMenu();
    
    // Update viewport height on resize
    window.addEventListener('resize', function() {
        setMobileViewportHeight();
    });
});

/**
 * Initialize mobile menu functionality
 */
function initializeMobileMenu() {
    const header = document.querySelector('.header');
    
    // Skip if header doesn't exist
    if (!header) return;
    
    // Create mobile menu toggle button
    const mobileMenuToggle = document.createElement('button');
    mobileMenuToggle.className = 'mobile-menu-toggle';
    mobileMenuToggle.innerHTML = `
        <span></span>
        <span></span>
        <span></span>
    `;
    
    // Create overlay
    const mobileNavOverlay = document.createElement('div');
    mobileNavOverlay.className = 'mobile-nav-overlay';
    
    // Add elements to the DOM
    header.insertBefore(mobileMenuToggle, header.firstChild);
    document.body.appendChild(mobileNavOverlay);
    
    // Get the main navigation
    const mainNav = document.querySelector('.main-nav');
    
    // Add click event listener to toggle menu
    mobileMenuToggle.addEventListener('click', function() {
        mobileMenuToggle.classList.toggle('active');
        mainNav.classList.toggle('active');
        mobileNavOverlay.classList.toggle('active');
        
        // Prevent body scrolling when menu is open
        if (mainNav.classList.contains('active')) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
    });
    
    // Close menu when clicking overlay
    mobileNavOverlay.addEventListener('click', function() {
        mobileMenuToggle.classList.remove('active');
        mainNav.classList.remove('active');
        mobileNavOverlay.classList.remove('active');
        document.body.style.overflow = '';
    });
    
    // Close menu when clicking a nav link
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            mobileMenuToggle.classList.remove('active');
            mainNav.classList.remove('active');
            mobileNavOverlay.classList.remove('active');
            document.body.style.overflow = '';
        });
    });
    
    // Close menu on window resize to desktop size
    window.addEventListener('resize', function() {
        if (window.innerWidth > 768 && mainNav.classList.contains('active')) {
            mobileMenuToggle.classList.remove('active');
            mainNav.classList.remove('active');
            mobileNavOverlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    });
}

/**
 * Sets the CSS variable --vh based on the actual viewport height
 * This fixes issues with viewport height on mobile browsers
 */
function setMobileViewportHeight() {
    // Get the viewport height and multiply by 1% to get a value for a vh unit
    let vh = window.innerHeight * 0.01;
    // Set the value in the --vh custom property to the root of the document
    document.documentElement.style.setProperty('--vh', `${vh}px`);
}

/**
 * Initializes the loading screen with logo and animation
 */
function initializeLoadingScreen() {
    // Create loading overlay
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    
    // Add logo
    const loadingLogo = document.createElement('img');
    loadingLogo.className = 'loading-logo';
    loadingLogo.src = '/static/images/luckytrainai-logo.png';
    loadingLogo.alt = 'LuckyTrainAI';
    
    // Add loading video
    const loadingVideo = document.createElement('video');
    loadingVideo.className = 'loading-video';
    loadingVideo.autoplay = true;
    loadingVideo.muted = true;
    loadingVideo.loop = false;
    
    // Create video source
    const videoSource = document.createElement('source');
    videoSource.src = '/static/media/loading-animation.mp4';
    videoSource.type = 'video/mp4';
    
    // Append elements
    loadingVideo.appendChild(videoSource);
    loadingOverlay.appendChild(loadingLogo);
    loadingOverlay.appendChild(loadingVideo);
    document.body.appendChild(loadingOverlay);
    
    // Handle video events
    loadingVideo.addEventListener('ended', function() {
        // Fade out the loading screen after video ends
        loadingOverlay.style.opacity = '0';
        setTimeout(function() {
            loadingOverlay.remove();
        }, 500);
    });
    
    // Fallback: if video doesn't load or play, remove loading screen after timeout
    setTimeout(function() {
        if (document.body.contains(loadingOverlay)) {
            loadingOverlay.style.opacity = '0';
            setTimeout(function() {
                loadingOverlay.remove();
            }, 500);
        }
    }, 5000);
}

/**
 * Initializes voice input and output features
 */
function initializeVoiceFeatures() {
    // Check if chat form exists (we're on a chat page)
    const chatForm = document.getElementById('chat-form');
    if (!chatForm) return;
    
    // Add voice reply button to the chat form
    const voiceReplyBtn = document.createElement('button');
    voiceReplyBtn.type = 'button';
    voiceReplyBtn.className = 'voice-reply-btn';
    voiceReplyBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="23"></line>
            <line x1="8" y1="23" x2="16" y2="23"></line>
        </svg>
    `;
    
    // Insert the button before the send button
    const sendButton = document.getElementById('send-button');
    chatForm.insertBefore(voiceReplyBtn, sendButton);
    
    // Voice recognition setup
    let recognition = null;
    try {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            
            // Configure language (auto-detect based on browser)
            recognition.lang = navigator.language || 'ru-RU';
        }
    } catch (e) {
        console.warn('Speech recognition not supported in this browser');
    }
    
    // Handle voice button click
    if (recognition) {
        voiceReplyBtn.addEventListener('click', function() {
            if (voiceReplyBtn.classList.contains('active')) {
                // Stop listening
                recognition.stop();
                voiceReplyBtn.classList.remove('active');
            } else {
                // Start listening
                recognition.start();
                voiceReplyBtn.classList.add('active');
            }
        });
        
        // Handle recognition results
        recognition.addEventListener('result', function(event) {
            const transcript = Array.from(event.results)
                .map(result => result[0])
                .map(result => result.transcript)
                .join('');
            
            // Update input field with transcript
            const messageInput = document.getElementById('message-input');
            messageInput.value = transcript;
            messageInput.focus();
            
            // Stop listening
            voiceReplyBtn.classList.remove('active');
        });
        
        // Handle recognition end
        recognition.addEventListener('end', function() {
            voiceReplyBtn.classList.remove('active');
        });
        
        // Handle recognition errors
        recognition.addEventListener('error', function(event) {
            console.error('Speech recognition error:', event.error);
            voiceReplyBtn.classList.remove('active');
        });
    } else {
        // If speech recognition is not supported, show a tooltip on hover
        voiceReplyBtn.title = 'Voice input is not supported in your browser';
        voiceReplyBtn.classList.add('disabled');
        voiceReplyBtn.addEventListener('click', function() {
            alert('Voice input is not supported in your browser');
        });
    }
    
    // Add text-to-speech for assistant responses
    setupTextToSpeech();
}

/**
 * Sets up text-to-speech functionality for assistant responses
 */
function setupTextToSpeech() {
    // Check if speech synthesis is supported
    if ('speechSynthesis' in window) {
        // Create button to toggle TTS for all messages
        const ttsSwitchContainer = document.createElement('div');
        ttsSwitchContainer.className = 'tts-switch-container';
        ttsSwitchContainer.innerHTML = `
            <label class="tts-switch">
                <input type="checkbox" id="tts-toggle">
                <span class="tts-slider"></span>
                <span class="tts-label">Voice Responses</span>
            </label>
        `;
        
        // Add to controls
        const chatControls = document.querySelector('.chat-controls');
        if (chatControls) {
            chatControls.appendChild(ttsSwitchContainer);
        }
        
        // Get the toggle element
        const ttsToggle = document.getElementById('tts-toggle');
        
        // Restore saved preference
        const ttsPref = localStorage.getItem('tts-enabled');
        if (ttsPref === 'true') {
            ttsToggle.checked = true;
        }
        
        // Save preference on change
        ttsToggle.addEventListener('change', function() {
            localStorage.setItem('tts-enabled', this.checked);
            
            // Stop any ongoing speech when toggled off
            if (!this.checked) {
                window.speechSynthesis.cancel();
            }
        });
        
        // Add buttons to each assistant message for individual TTS
        const messages = document.querySelectorAll('.message.assistant');
        addTtsButtonsToMessages(messages);
        
        // Set up observer to add TTS buttons to new messages
        const chatMessages = document.querySelector('.chat-messages');
        if (chatMessages) {
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.addedNodes.length) {
                        mutation.addedNodes.forEach(function(node) {
                            if (node.classList && node.classList.contains('message') && node.classList.contains('assistant')) {
                                addTtsButtonToMessage(node);
                                
                                // Auto-play TTS if enabled
                                if (ttsToggle.checked) {
                                    setTimeout(function() {
                                        // Only play if it's the last message
                                        if (node === chatMessages.lastChild) {
                                            const content = node.querySelector('.message-content p');
                                            if (content) {
                                                playTTS(content.textContent);
                                            }
                                        }
                                    }, 500);
                                }
                            }
                        });
                    }
                });
            });
            
            observer.observe(chatMessages, { childList: true });
        }
    }
}

/**
 * Adds TTS buttons to multiple messages
 * @param {NodeList} messages - The assistant messages to add buttons to
 */
function addTtsButtonsToMessages(messages) {
    messages.forEach(addTtsButtonToMessage);
}

/**
 * Adds a TTS button to a single message
 * @param {Element} message - The assistant message to add a button to
 */
function addTtsButtonToMessage(message) {
    // Check if the message already has a TTS button
    if (message.querySelector('.tts-btn')) return;
    
    const messageContent = message.querySelector('.message-content');
    if (!messageContent) return;
    
    const ttsBtn = document.createElement('button');
    ttsBtn.className = 'tts-btn';
    ttsBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
        </svg>
    `;
    
    // Add event listener to play TTS
    ttsBtn.addEventListener('click', function() {
        const content = messageContent.querySelector('p');
        if (content) {
            playTTS(content.textContent);
        }
    });
    
    // Add the button to the message content
    const feedbackDiv = messageContent.querySelector('.message-feedback');
    if (feedbackDiv) {
        messageContent.insertBefore(ttsBtn, feedbackDiv);
    } else {
        messageContent.appendChild(ttsBtn);
    }
}

/**
 * Plays text-to-speech for the given text
 * @param {string} text - The text to speak
 */
function playTTS(text) {
    // Stop any ongoing speech
    window.speechSynthesis.cancel();
    
    // Create new speech synthesis utterance
    const utterance = new SpeechSynthesisUtterance(text);
    
    // Set language (auto-detect based on content or default to Russian)
    const langPref = localStorage.getItem('tts-language') || navigator.language || 'ru-RU';
    utterance.lang = langPref;
    
    // Get voices
    let voices = window.speechSynthesis.getVoices();
    
    // If voices aren't loaded yet, wait and try again
    if (voices.length === 0) {
        window.speechSynthesis.addEventListener('voiceschanged', function() {
            voices = window.speechSynthesis.getVoices();
            setVoice();
        });
    } else {
        setVoice();
    }
    
    function setVoice() {
        // Find an appropriate voice
        const preferredVoice = voices.find(voice => voice.lang.startsWith(langPref.slice(0, 2)));
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }
    }
    
    // Set other properties
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    
    // Speak
    window.speechSynthesis.speak(utterance);
}

/**
 * Initializes dynamic color effects throughout the UI
 */
function initializeColorEffects() {
    // Add grid overlay for cyberpunk effect
    const gridOverlay = document.createElement('div');
    gridOverlay.className = 'grid-overlay';
    document.body.appendChild(gridOverlay);
    
    // Add hover effects for interactive elements
    addHoverEffects();
    
    // Add pulsing effect to important elements
    addPulsingEffects();
    
    // Add animated backgrounds
    addAnimatedBackgrounds();
}

/**
 * Adds hover effects to interactive elements
 */
function addHoverEffects() {
    // Add hover effects to buttons on hover
    document.querySelectorAll('button, .nav-link').forEach(element => {
        element.addEventListener('mouseenter', function() {
            this.style.transition = 'all 0.3s ease';
            
            // Create ripple effect
            if (!this.querySelector('.ripple-effect')) {
                const ripple = document.createElement('span');
                ripple.className = 'ripple-effect';
                this.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 1000);
            }
        });
    });
}

/**
 * Adds pulsing effects to important UI elements
 */
function addPulsingEffects() {
    // Add pulsing effect to logo
    const logo = document.querySelector('.logo');
    if (logo) {
        logo.style.animation = 'pulse-logo 3s infinite alternate';
    }
    
    // Add subtle pulsing to chat input
    const chatInput = document.getElementById('message-input');
    if (chatInput) {
        chatInput.addEventListener('focus', function() {
            this.style.animation = 'pulse-input 2s infinite alternate';
        });
        
        chatInput.addEventListener('blur', function() {
            this.style.animation = 'none';
        });
    }
}

/**
 * Adds animated backgrounds to containers
 */
function addAnimatedBackgrounds() {
    // Add animated gradient to header
    const header = document.querySelector('.header');
    if (header) {
        const headerGlow = document.createElement('div');
        headerGlow.className = 'header-glow';
        header.appendChild(headerGlow);
    }
    
    // Add particle effect to background
    createParticleEffect();
}

/**
 * Creates a particle effect in the background
 */
function createParticleEffect() {
    const canvas = document.createElement('canvas');
    canvas.className = 'particle-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '-1';
    document.body.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    
    // Create particles
    const particles = [];
    const particleCount = 30;
    const colors = ['#00FFE5', '#9933FF', '#FF0033', '#00FF66'];
    
    function createParticles() {
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                radius: Math.random() * 2 + 1,
                color: colors[Math.floor(Math.random() * colors.length)],
                speedX: Math.random() * 0.5 - 0.25,
                speedY: Math.random() * 0.5 - 0.25,
                alpha: Math.random() * 0.5 + 0.2
            });
        }
    }
    
    // Animate particles
    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = p.color + Math.floor(p.alpha * 255).toString(16).padStart(2, '0');
            ctx.fill();
            
            // Update position
            p.x += p.speedX;
            p.y += p.speedY;
            
            // Wrap around edges
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
        }
        
        requestAnimationFrame(animateParticles);
    }
    
    // Initialize canvas and particles
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    createParticles();
    animateParticles();
}

/**
 * Handles the response loading animation with custom video
 * @param {boolean} isLoading - Whether the response is loading
 */
function handleResponseLoading(isLoading) {
    // Remove existing loading indicator if any
    const existingIndicator = document.querySelector('.response-loading');
    if (existingIndicator) {
        existingIndicator.remove();
    }
    
    if (isLoading) {
        // Create and add the loading indicator with video
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'response-loading';
        
        const loadingVideo = document.createElement('video');
        loadingVideo.autoplay = true;
        loadingVideo.loop = true;
        loadingVideo.muted = true;
        loadingVideo.width = 240;
        
        const videoSource = document.createElement('source');
        videoSource.src = '/static/media/response-loading.mp4';
        videoSource.type = 'video/mp4';
        
        loadingVideo.appendChild(videoSource);
        loadingIndicator.appendChild(loadingVideo);
        
        // Add to the chat messages container
        const chatMessages = document.querySelector('.chat-messages');
        if (chatMessages) {
            chatMessages.appendChild(loadingIndicator);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
}

// Export functions for use in other scripts
window.LuckyTrainAI = {
    playTTS,
    handleResponseLoading
}; 