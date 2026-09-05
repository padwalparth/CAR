import { useState, useEffect, useRef, useCallback } from 'react';
import { LiveMonitorWebSocket, ConnectionState, LiveEventMessage } from '../services/api/websocket';

export function useLiveMonitor() {
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [events, setEvents] = useState<LiveEventMessage[]>([]);
  const socketRef = useRef<LiveMonitorWebSocket | null>(null);

  const connectStream = useCallback(() => {
    if (!socketRef.current) {
      socketRef.current = new LiveMonitorWebSocket();
      socketRef.current.onStateChange(setConnectionState);
      socketRef.current.onEvent((evt) => {
        setEvents(prev => [evt, ...prev].slice(0, 100)); // Keep max 100 recent events
      });
    }
    socketRef.current.connect();
  }, []);

  const disconnectStream = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
    }
  }, []);

  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  return {
    connectionState,
    events,
    connectStream,
    disconnectStream,
    clearEvents: () => setEvents([])
  };
}
