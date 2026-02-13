import { API_BASE_URL } from '../config';

export function AgentWidget() {
  return `
    <!-- Large Horizontal Dashboard Container -->
    <div class="relative w-full max-w-[1100px] h-[700px] rounded-2xl overflow-hidden shadow-2xl border border-brand-gold/10 flex bg-[#0a0a0a] mx-auto my-8 font-sans">
      
      <img src="/bg-aureeq-option3-seamless.png" class="absolute inset-0 w-full h-full object-cover opacity-70 z-0" alt="Background" />
      


      <!-- Luxury Gradient Overlays -->
      <div class="absolute inset-0 bg-gradient-to-r from-black/40 via-transparent to-black/20 pointer-events-none z-10"></div>
      <div class="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/60 pointer-events-none z-10"></div>
      
      <!-- 1. Left Side: Chat Interface (Full width for scrollbar) -->
      <div class="w-full flex flex-col relative z-30 pointer-events-none">
        <!-- Header (Back to pointer-events-auto for interactions) -->
        <div class="h-14 flex items-center justify-between px-6 bg-black/20 backdrop-blur-sm border-b border-white/5 pointer-events-auto relative">
          <img src="/aureeq-logo-text.png" class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 h-[88px] object-contain opacity-90" alt="Aureeq" />
          <div class="flex items-center gap-3">
            <span class="text-white font-medium text-sm tracking-wide">Aureeq Assistant</span>
          </div>
          <!-- Logout / Reset Identity Button -->
          <button id="logout-btn" class="text-white/50 hover:text-red-400 transition-colors z-50 pointer-events-auto" title="Reset Identity">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path></svg>
          </button>
        </div>

        <!-- Chat Messages Area (Full width, pointer-events-auto) -->
        <style>
          #chat-messages::-webkit-scrollbar {
            width: 4px;
          }
          #chat-messages::-webkit-scrollbar-track {
            background: transparent;
          }
          #chat-messages::-webkit-scrollbar-thumb {
            background: rgba(212, 175, 55, 0.3);
            border-radius: 10px;
          }
          #chat-messages::-webkit-scrollbar-thumb:hover {
            background: rgba(212, 175, 55, 0.6);
          }
        </style>
        <div class="flex-1 overflow-y-auto p-8 pr-[30%] flex flex-col gap-6 scroll-smooth pointer-events-auto" id="chat-messages">
          <!-- AI Welcome Message -->
          <div class="animate-fade-in group max-w-[85%]">
             <div class="bg-brand-gold text-black text-[11px] font-bold px-4 py-1 rounded-t-xl w-full tracking-[0.2em] uppercase">
               Aureeq
             </div>
             <div class="bg-[#1a1a1a] border-x border-b border-white/5 text-slate-100 text-[15px] px-5 py-3 rounded-b-xl shadow-2xl leading-[1.5] font-normal font-inter">
               Hello I am AUREEQ your personal assistant, How may I help you today?
             </div>
          </div>
        </div>

        <!-- Footer: Input Area (Pointer-events-auto) -->
        <div class="h-24 px-8 pb-6 flex items-end gap-3 z-30 pointer-events-auto pr-[30%]">
          <button id="mic-btn" class="flex w-12 h-12 rounded-xl bg-brand-black/40 border border-brand-gold/20 items-center justify-center text-brand-gold hover:bg-brand-gold hover:text-black transition-all group shrink-0 backdrop-blur-md">
            <svg id="mic-icon" class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 3.01-2.55 5.5-5.5 5.5S6 14.01 6 11H4c0 3.53 2.61 6.43 6 6.92V21h4v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
          </button>

          <div class="flex-1 relative">
            <input id="chat-input" type="text" placeholder="Write message here..." autocomplete="off"
              class="w-full bg-brand-black/60 border border-white/10 rounded-xl h-12 px-5 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-brand-gold/40 transition-colors shadow-2xl backdrop-blur-md" />
          </div>

          <button id="send-btn" class="w-12 h-12 rounded-xl bg-brand-gold/80 flex items-center justify-center shadow-lg shadow-brand-gold/20 hover:bg-brand-gold transition-all group shrink-0 backdrop-blur-md">
            <svg class="w-5 h-5 text-black group-hover:translate-x-0.5 transition-transform" fill="currentColor" viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
          </button>
        </div>
      </div>

      <!-- 2. Right Side: Avatar (30% absolute) -->
      <div class="absolute top-0 right-0 bottom-0 w-[30%] overflow-hidden flex flex-col z-20 pointer-events-none">
        <!-- 3D Canvas Container -->
        <div class="absolute inset-0 flex items-end justify-center pointer-events-none translate-y-12">
          <canvas id="avatar-canvas" class="w-full h-full object-contain pointer-events-none"></canvas>
        </div>
      </div>

      <!-- 3. Mandatory Onboarding Modal -->
      <div id="onboarding-modal" class="absolute inset-0 z-[100] bg-black/80 backdrop-blur-md flex items-center justify-center hidden">
        <div class="max-w-[400px] w-full bg-[#111] border border-brand-gold/20 rounded-2xl p-8 shadow-[0_0_50px_rgba(212,175,55,0.15)] animate-fade-in">
          <div class="text-center mb-6">
            <div class="text-brand-gold text-4xl font-black mb-2 tracking-tighter">AUREEQ</div>
            <p class="text-white/60 text-xs tracking-[0.2em] font-bold uppercase">Sales Strategist Onboarding</p>
          </div>
          
          <div class="space-y-4">
            <div>
              <label class="block text-brand-gold/60 text-[10px] font-bold uppercase tracking-widest mb-1 ml-1 text-left">Full Name</label>
              <input id="ob-name" type="text" placeholder="Enter your name" class="w-full bg-white/5 border border-white/10 rounded-xl h-12 px-5 text-white text-sm focus:outline-none focus:border-brand-gold/40 transition-all shadow-inner" />
            </div>
            <div>
              <label class="block text-brand-gold/60 text-[10px] font-bold uppercase tracking-widest mb-1 ml-1 text-left">Email Address</label>
              <input id="ob-email" type="email" placeholder="email@example.com" class="w-full bg-white/5 border border-white/10 rounded-xl h-12 px-5 text-white text-sm focus:outline-none focus:border-brand-gold/40 transition-all shadow-inner" />
            </div>
            
            <button id="ob-submit" class="w-full h-12 bg-brand-gold text-black font-black text-sm rounded-xl shadow-lg shadow-brand-gold/20 hover:scale-[1.02] active:scale-[0.98] transition-all uppercase tracking-widest mt-4">
              Access Strategist
            </button>
          </div>
          
          <p class="text-white/30 text-[9px] text-center mt-6 uppercase tracking-tight">Your data is synced securely to the Aureeq persistence layer.</p>
        </div>
      </div>

    </div>
  `;
}

export function setupAgentInteraction(avatarRenderer) {
  // Session is now persisted across reloads
  // localStorage.removeItem('aureeq_user_name');
  // localStorage.removeItem('aureeq_user_email');

  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const messagesContainer = document.getElementById('chat-messages');

  // Logic for Logout Button
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      if (confirm("Reset identity and show onboarding again?")) {
        localStorage.removeItem('aureeq_user_name');
        localStorage.removeItem('aureeq_user_email');
        localStorage.removeItem('aureeq_user_prefs');
        location.reload(); // Reload to trigger onboarding check
      }
    });
  }

  // --- ORDER POPUP UI INJECTION ---
  const modalHtml = `
    <div id="order-modal" class="hidden absolute inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center p-6 animate-fade-in pointer-events-auto">
        <div class="bg-[#1a1a1a] border border-brand-gold/30 rounded-2xl p-6 w-full max-w-sm shadow-2xl transform scale-100 transition-all">
            
            <div class="flex flex-col items-center gap-4 text-center">
                <div class="w-16 h-16 bg-brand-gold/10 rounded-full flex items-center justify-center text-3xl mb-2">
                    🍽️
                </div>
                
                <div>
                    <h3 class="text-white text-lg font-bold mb-1">Confirm Order</h3>
                    <p class="text-slate-400 text-sm">Would you like to add this to your cart?</p>
                </div>

                <div class="bg-white/5 border border-white/5 rounded-xl p-4 w-full flex justify-between items-center">
                    <span id="modal-item-name" class="text-white font-medium">Item Name</span>
                    <span id="modal-item-price" class="text-brand-gold font-bold">£0.00</span>
                </div>

                <div class="flex gap-3 w-full mt-2">
                    <button id="modal-cancel-btn" class="flex-1 bg-white/5 hover:bg-white/10 text-slate-300 py-3 rounded-xl font-medium transition-colors">
                        Cancel
                    </button>
                    <button id="modal-confirm-btn" class="flex-1 bg-brand-gold hover:bg-yellow-400 text-black py-3 rounded-xl font-bold transition-colors shadow-lg">
                        Add to Cart
                    </button>
                </div>
            </div>
        </div>
    </div>`;

  // Inject if not present
  const widgetContainer = document.querySelector('.relative.w-full.max-w-\\[1100px\\]');
  if (widgetContainer && !document.getElementById('order-modal')) {
    const temp = document.createElement('div');
    temp.innerHTML = modalHtml;
    widgetContainer.appendChild(temp.firstElementChild);
  }

  const orderModal = document.getElementById('order-modal');
  const modalName = document.getElementById('modal-item-name');
  const modalPrice = document.getElementById('modal-item-price');
  const modalConfirm = document.getElementById('modal-confirm-btn');
  const modalCancel = document.getElementById('modal-cancel-btn');

  const showOrderPopup = (name, price) => {
    if (!orderModal) return;
    modalName.textContent = name;
    modalPrice.textContent = price;
    orderModal.classList.remove('hidden');

    // Clean listeners first (crudely by cloning or assigning new onclick)
    modalConfirm.onclick = () => {
      console.log("Adding to cart:", name);
      // Post Message to Parent (Website)
      window.parent.postMessage({ type: 'AUREEQ_CART_ADD', detail: { query: name, price: price } }, '*');

      // Also sync with backend
      const user = getStoredUser();
      fetch(`${API_BASE_URL}/order`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: user.email, items: [name], total: parseFloat(price.replace(/[^0-9.]/g, '')) })
      });

      addMessage(`✅ **${name}** added to cart!`, false);
      orderModal.classList.add('hidden');
    };

    modalCancel.onclick = () => {
      orderModal.classList.add('hidden');
    };
  };

  if (!input || !sendBtn) return;

  const addMessage = (text, isUser = false) => {
    const div = document.createElement('div');
    div.className = "animate-fade-in mb-4";

    if (isUser) {
      div.innerHTML = `
                <div class="flex justify-end">
                    <div class="max-w-[85%]">
                        <div class="bg-[#1a1a1a] text-white text-[11px] font-bold px-4 py-1 rounded-t-xl w-full tracking-[0.2em] uppercase">User</div>
                        <div class="bg-brand-gold text-black text-[15px] px-5 py-3 rounded-b-xl shadow-2xl leading-[1.5] font-normal font-inter">${text}</div>
                    </div>
                </div>`;
    } else {
      let displayText = text;
      const orderMatch = text.match(/\[ORDER:\s*(.*?)\s*\|\s*(.*?)\]/i);

      if (orderMatch) {
        const name = orderMatch[1].trim();
        const price = orderMatch[2].trim();
        displayText = text.replace(orderMatch[0], '').trim();

        // Trigger popup
        setTimeout(() => showOrderPopup(name, price), 800);
      }

      // Support Markdown links in static messages
      displayText = displayText
        .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" class="text-brand-gold underline hover:text-yellow-400 break-all">$1</a>');

      div.innerHTML = `
                <div class="max-w-[85%]">
                  <div class="bg-brand-gold text-black text-[11px] font-bold px-4 py-1 rounded-t-xl w-full tracking-[0.2em] uppercase">Aureeq</div>
                  <div class="bg-[#1a1a1a] border-x border-b border-white/5 text-slate-100 text-[15px] px-5 py-3 rounded-b-xl shadow-2xl leading-[1.5] font-normal font-inter msg-content">${displayText}</div>
                </div>`;
    }
    messagesContainer.appendChild(div);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return div;
  };

  const getStoredUser = () => ({
    name: localStorage.getItem('aureeq_user_name') || 'Guest',
    email: localStorage.getItem('aureeq_user_email') || null,
    preferences: localStorage.getItem('aureeq_user_prefs') || ''
  });

  let isVoiceInputSource = false;

  const handleSend = async () => {
    const text = input.value.trim();
    if (!text) return;

    const wasVoice = isVoiceInputSource;
    isVoiceInputSource = false;

    // Detect Identity in text if needed
    // (omitted simple identify logic for brevity, main focus is send)

    addMessage(text, true);
    input.value = '';

    const user = getStoredUser();

    import('../lib/salesAgent').then(async ({ SalesAgent }) => {
      // Assemble context using RAG/etc on client side helper if needed, 
      // but here we just pass text to backend which handles heavy lifting
      const context = await SalesAgent.assembleContext(text, user);

      let aiMsgDiv = addMessage("Aureeq is thinking...");
      let contentEl = aiMsgDiv.querySelector('.msg-content');

      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, user_id: user.email, context: context, user_metadata: user })
      });

      if (!res.ok) {
        const errorText = await res.text();
        contentEl.innerText = `Error: Backend returned ${res.status}. ${errorText}`;
        console.error("Chat Request Failed:", res.status, errorText);
        return;
      }

      let fullText = "";
      try {
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let hasTokens = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            console.log("Stream complete. Full text length:", fullText.length);
            break;
          }

          const chunk = decoder.decode(value, { stream: true });
          console.log("RX Chunk:", chunk);

          if (!hasTokens) {
            // Check if chunk has visible content (ignore ZWSP \u200B and whitespace)
            const isVisible = chunk.replace(/[\u200B\s]/g, '').length > 0;
            if (!isVisible) continue;

            contentEl.textContent = "";
            hasTokens = true;
          }

          fullText += chunk;

          // Simple Markdown Link Converter: [text](url) -> <a href="url" target="_blank">text</a>
          const safeText = fullText
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\n/g, "<br/>")
            .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" class="text-brand-gold underline hover:text-yellow-400 break-all">$1</a>');

          contentEl.innerHTML = safeText;
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        if (!hasTokens || !fullText.trim()) {
          contentEl.textContent = "The brain is cooling down. Please try again.";
        }
      } catch (streamError) {
        console.error("Stream interrupted:", streamError);
        contentEl.innerText += "\n[Connection interrupted. Please try again.]";
      }

      // Trigger TTS? (Only if it was a voice input - User Request)
      if (avatarRenderer && fullText.trim() && wasVoice) {
        // Always play audio if avatar is present
        try {
          const ttsRes = await fetch(`${API_BASE_URL}/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: fullText })
          });
          const ttsData = await ttsRes.json();
          if (ttsData.audio_url) {
            const path = ttsData.audio_url;
            // Fix: If path already starts with /api (from backend), don't prepend API_BASE_URL again
            const fullAudioUrl = (path.startsWith('http') || path.startsWith('/')) ? path : `${API_BASE_URL}${path}`;
            avatarRenderer.speakFromUrl(fullAudioUrl);
          }
        } catch (e) {
          console.error("TTS Error:", e);
        }
      }

      // Check order tag again in full text for safety
      const orderMatch = fullText.match(/\[ORDER:\s*(.*?)\s*\|\s*(.*?)\]/i);
      if (orderMatch) {
        const name = orderMatch[1].trim();
        const price = orderMatch[2].trim();
        setTimeout(() => showOrderPopup(name, price), 500);
      }

    });
  };

  sendBtn.onclick = handleSend;
  input.onkeypress = (e) => { if (e.key === 'Enter') handleSend(); };

  // --- STRICT MIC LOGIC ---
  const micBtn = document.getElementById('mic-btn');
  const micIcon = document.getElementById('mic-icon');
  let recognition = null;
  let isRecording = false;

  const initRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return null;
    const rec = new SpeechRecognition();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = 'en-US';

    rec.onresult = (e) => {
      let fullTranscript = '';
      for (let i = 0; i < e.results.length; ++i) {
        fullTranscript += e.results[i][0].transcript;
      }
      input.value = fullTranscript;
      isVoiceInputSource = true;
    };

    rec.onend = () => {
      if (isRecording) {
        isRecording = false;
        micBtn.classList.remove('bg-red-500', 'animate-pulse');
        micIcon.innerHTML = '<path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 3.01-2.55 5.5-5.5 5.5S6 14.01 6 11H4c0 3.53 2.61 6.43 6 6.92V21h4v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>';
      }
    };

    return rec;
  };

  micBtn.onclick = async () => {
    // 1. If currently recording -> STOP and SEND
    if (isRecording) {
      if (recognition) {
        recognition.stop();
        isRecording = false;
        micBtn.classList.remove('bg-red-500', 'animate-pulse');
        micIcon.innerHTML = '<path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 3.01-2.55 5.5-5.5 5.5S6 14.01 6 11H4c0 3.53 2.61 6.43 6 6.92V21h4v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>'; // Mic Icon

        // Auto-send after a small delay to catch the last word
        setTimeout(() => {
          if (input.value.trim().length > 0) handleSend();
        }, 300);
      }
    } else {
      // 2. If NOT recording -> ASK PERMISSION & START
      try {
        await navigator.mediaDevices.getUserMedia({ audio: true });

        if (!recognition) recognition = initRecognition();
        if (recognition) {
          input.value = ''; // Clear previous text for a fresh start
          recognition.start();
          isRecording = true;
          micBtn.classList.add('bg-red-500', 'animate-pulse');
          // Change icon to Square (Stop)
          micIcon.innerHTML = '<rect x="6" y="6" width="12" height="12" />';
        } else {
          alert("Speech Recognition not supported in this browser.");
        }
      } catch (e) {
        console.error("Mic Permission Denied or Error:", e);
        alert("Microphone permission is required to use voice input.");
      }
    }
  };

  // Onboarding Logic (Simplified for check)
  const modal = document.getElementById('onboarding-modal');
  const checkOnboarding = () => {
    const user = getStoredUser();
    if (!user.email || !user.name) {
      if (modal) modal.classList.remove('hidden');
      return false;
    }
    if (modal) modal.classList.add('hidden');
    return true;
  };

  const obSubmit = document.getElementById('ob-submit');
  if (obSubmit) {
    obSubmit.onclick = () => {
      const name = document.getElementById('ob-name').value;
      const email = document.getElementById('ob-email').value;
      if (name && email) {
        localStorage.setItem('aureeq_user_name', name);
        localStorage.setItem('aureeq_user_email', email);
        checkOnboarding();
        initWelcome(name);
      }
    };
  }

  const isOnboarded = checkOnboarding();
  const initWelcome = async (name) => {
    try {
      const user = getStoredUser();
      const userName = name || user.name || "Guest";

      console.log("Initializing welcome for:", userName);

      const res = await fetch(`${API_BASE_URL}/welcome?name=${encodeURIComponent(userName)}&user_id=${encodeURIComponent(user.email || "")}`);
      const data = await res.json();

      if (data.response) {
        // Optional: Update initial chat message if you want dynamic text
        // For now, we keep the static one or we could find and update it
        // const welcomeMsg = document.querySelector('#chat-messages > div:first-child .text-slate-100');
        // if (welcomeMsg) welcomeMsg.textContent = data.response;
      }

      if (data.audio_url) {
        console.log("Playing welcome audio with lip-sync:", data.audio_url);
        const path = data.audio_url;
        // Fix: If path already starts with /api or / (from backend), don't prepend API_BASE_URL again
        const fullWelcomeUrl = (path.startsWith('http') || path.startsWith('/')) ? path : `${API_BASE_URL}${path}`;

        if (window.avatarFunctions && window.avatarFunctions.speakFromUrl) {
          window.avatarFunctions.speakFromUrl(fullWelcomeUrl);
        } else {
          // Fallback if avatar not ready
          const audio = new Audio(fullWelcomeUrl);
          audio.play().catch(e => console.error("Auto-play failed:", e));
        }
      }
    } catch (e) {
      console.error("Welcome init failed:", e);
    }
  };

  if (isOnboarded) {
    // Small delay to ensure everything is loaded
    setTimeout(() => initWelcome(), 1000);
  }
}
