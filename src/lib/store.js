export class Store {
    constructor(initialState) {
        this.state = initialState;
        this.listeners = [];
    }

    subscribe(listener) {
        this.listeners.push(listener);
        return () => {
            this.listeners = this.listeners.filter(l => l !== listener);
        };
    }

    setState(newState) {
        this.state = { ...this.state, ...newState };
        this.listeners.forEach(listener => listener(this.state));
    }

    getState() {
        return this.state;
    }
}

export const appStore = new Store({
    view: 'home', // home, cart, product
    cart: [],
    isAgentActive: false
});
