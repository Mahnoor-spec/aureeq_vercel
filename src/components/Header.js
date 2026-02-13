import { appStore } from '../lib/store';

export function Header() {
    const state = appStore.getState();
    const cartCount = state.cart.length;

    return `
    <header class="w-full h-20 flex items-center justify-between px-6 md:px-12 border-b border-glass-border bg-glass-surface backdrop-blur-md sticky top-0 z-50">
      <!-- Logo -->
      <div class="flex items-center gap-3 cursor-pointer group" onclick="window.navigate('home')">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-lg shadow-primary/20 group-hover:shadow-primary/40 transition-all duration-300">
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
        </div>
        <span class="text-2xl font-display font-bold text-white tracking-wide">AUREEQ</span>
      </div>

      <!-- Actions -->
      <div class="flex items-center gap-6">
        <button class="hidden md:flex items-center gap-2 text-slate-400 hover:text-white transition-colors duration-200 text-sm font-medium">
          <span>Search</span>
          <kbd class="px-2 py-0.5 rounded bg-slate-800 text-xs text-slate-500">/</kbd>
        </button>
        
        <div class="h-6 w-px bg-slate-800"></div>

        <button class="relative p-2 rounded-lg hover:bg-slate-800/50 transition-colors group">
          <svg class="w-6 h-6 text-slate-300 group-hover:text-white transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
          ${cartCount > 0 ? `<span class="absolute top-0 right-0 w-5 h-5 rounded-full bg-accent text-white text-xs flex items-center justify-center font-bold shadow-lg shadow-accent/40 animate-pulse">${cartCount}</span>` : ''}
        </button>

        <button class="w-10 h-10 rounded-full bg-gradient-to-r from-slate-800 to-slate-700 border border-slate-600 flex items-center justify-center text-sm font-bold text-white shadow-lg overflow-hidden">
          <img src="https://ui-avatars.com/api/?name=User&background=random" class="w-full h-full object-cover opacity-80 hover:opacity-100 transition-opacity" />
        </button>
      </div>
    </header>
  `;
}
