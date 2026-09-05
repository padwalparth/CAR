export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface LiveEventMessage {
  id: string;
  timestamp: string;
  type: 'object' | 'pothole' | 'segmentation' | 'alert';
  message: string;
  data?: any;
}

export class LiveMonitorWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private stateListeners: Set<(state: ConnectionState) => void> = new Set();
  private eventListeners: Set<(event: LiveEventMessage) => void> = new Set();
  private reconnectTimer: any = null;

  constructor(url?: string) {
    this.url = url || import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/live';
  }

  public connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    this.notifyState('connecting');

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.notifyState('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          this.eventListeners.forEach(listener => listener(parsed));
        } catch {
          // Ignored non-JSON messages
        }
      };

      this.ws.onerror = () => {
        this.notifyState('error');
      };

      this.ws.onclose = () => {
        this.notifyState('disconnected');
      };
    } catch {
      this.notifyState('error');
    }
  }

  public disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.notifyState('disconnected');
  }

  public onStateChange(listener: (state: ConnectionState) => void): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  public onEvent(listener: (event: LiveEventMessage) => void): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  private notifyState(state: ConnectionState): void {
    this.stateListeners.forEach(listener => listener(state));
  }
}
