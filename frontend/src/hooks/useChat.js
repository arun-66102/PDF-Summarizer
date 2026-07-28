import { useState, useCallback } from 'react';

/**
 * useChat — manages chat state for RouteX.
 *
 * Each message: {
 *   id:       string,
 *   role:     "user" | "assistant",
 *   time:     string,
 *   text?:    string,      // user text input
 *   docName?: string,      // user attached PDF name
 *   result?:  object,      // assistant processing result
 *   source?:  "pdf"|"text" // what was processed
 * }
 */
export function useChat() {
  const [messages, setMessages] = useState([]);
  const [attachedDocs, setAttachedDocs] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState({ progress: 0, status: '', done: false });

  const addUserMessage = useCallback(({ text, docName }) => {
    const msg = {
      id: crypto.randomUUID(),
      role: 'user',
      time: new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true }),
      text: text || null,
      docName: docName || null,
    };
    setMessages((prev) => [...prev, msg]);
    return msg;
  }, []);

  const addAssistantMessage = useCallback((result, source) => {
    const msg = {
      id: crypto.randomUUID(),
      role: 'assistant',
      time: new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true }),
      result,
      source,
    };
    setMessages((prev) => [...prev, msg]);
    return msg;
  }, []);

  const attachDoc = useCallback((file) => {
    setAttachedDocs((prev) => {
      // Prevent duplicates
      if (prev.some((d) => d.name === file.name)) return prev;
      return [...prev, { name: file.name, size: file.size, file }];
    });
  }, []);

  const removeDoc = useCallback((index) => {
    setAttachedDocs((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const clearDocs = useCallback(() => {
    setAttachedDocs([]);
  }, []);

  const newChat = useCallback(() => {
    setMessages([]);
    setAttachedDocs([]);
    setProcessing(false);
    setProgress({ progress: 0, status: '', done: false });
  }, []);

  return {
    messages,
    attachedDocs,
    processing,
    progress,
    setProcessing,
    setProgress,
    addUserMessage,
    addAssistantMessage,
    attachDoc,
    removeDoc,
    clearDocs,
    newChat,
  };
}
