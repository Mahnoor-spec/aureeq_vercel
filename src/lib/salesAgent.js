
// salesAgent.js - The "Smart" logic for the frontend

import { API_BASE_URL } from '../config';

const API_BASE = API_BASE_URL;

// --- Regex Patterns for User Info Extraction ---
const REGEX_PATTERNS = {
    name: /my name is ([\w']+[\s\w']{0,20})|i am ([\w']+[\s\w']{0,20})|call me ([\w']+[\s\w']{0,20})/i,
    email: /([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/i,
    // Basic address detection (very loose)
    address: /my address is (.*)|deliver to (.*)|i live at (.*)/i,
    // Basic keyword detection for allergies
    allergy: /allergic to ([\w\s,]+)|sensitivity to ([\w\s,]+)/i,
};

// --- Storage Helper ---
const Storage = {
    get: (key) => {
        try {
            return JSON.parse(localStorage.getItem(`aureeq_${key}`));
        } catch {
            return localStorage.getItem(`aureeq_${key}`);
        }
    },
    set: (key, value) => {
        if (typeof value === 'object') {
            localStorage.setItem(`aureeq_${key}`, JSON.stringify(value));
        } else {
            localStorage.setItem(`aureeq_${key}`, value);
        }
    }
};

// --- Smart Context Assembly ---

async function fetchProductContext(query) {
    try {
        const res = await fetch(`${API_BASE}/products/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        const data = await res.json();
        return data.results.map(r => r.content).join('\n');
    } catch (e) {
        console.error("Product Search Failed:", e);
        return "";
    }
}

async function fetchOrderHistory(userId) {
    if (!userId) return "";
    try {
        const res = await fetch(`${API_BASE}/dataHandler?type=orders&user_id=${encodeURIComponent(userId)}`);
        const data = await res.json();
        if (data.orders && data.orders.length > 0) {
            return "PAST ORDERS:\n" + data.orders.map(o => `- ${o.items} (${o.created_at})`).join('\n');
        }
        return "";
    } catch (e) {
        console.error("Order Fetch Failed:", e);
        return "";
    }
}

export const SalesAgent = {

    extractUserInfo: (message) => {
        let extracted = {};

        // Name
        const nameMatch = message.match(REGEX_PATTERNS.name);
        if (nameMatch) {
            const rawName = nameMatch[1] || nameMatch[2] || nameMatch[3];
            if (rawName && rawName.split(' ').length <= 4) { // Safety check
                extracted.name = rawName.trim();
                Storage.set('user_name', extracted.name);
            }
        }

        // Email
        const emailMatch = message.match(REGEX_PATTERNS.email);
        if (emailMatch) {
            extracted.email = emailMatch[1].trim();
            Storage.set('user_email', extracted.email);
        }

        // Allergies (Append to existing)
        const allergyMatch = message.match(REGEX_PATTERNS.allergy);
        if (allergyMatch) {
            const allergy = allergyMatch[1] || allergyMatch[2];
            const currentPrefs = Storage.get('user_prefs') || "";
            const newPrefs = currentPrefs ? `${currentPrefs}, Allergic to ${allergy}` : `Allergic to ${allergy}`;
            Storage.set('user_prefs', newPrefs);
            extracted.preferences = newPrefs;
        }

        return extracted;
    },

    assembleContext: async (message, user) => {
        let contextParts = [];

        // 1. Data: Order History (only if user identified)
        const userId = user.email || user.name;
        if (userId && userId !== 'Guest') {
            const orderHistory = await fetchOrderHistory(userId);
            if (orderHistory) {
                contextParts.push(`\n${orderHistory}`);
            }
        }

        // 3. User Info (Explicit from Storage)
        const prefs = Storage.get('user_prefs');
        if (prefs) {
            contextParts.push(`\nUSER PREFERENCES: ${prefs}`);
        }

        return contextParts.join('\n');
    }
};
