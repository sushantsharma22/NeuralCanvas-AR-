"""Voice controller for NeuralCanvas AR."""

import speech_recognition as sr
import threading
import queue
import time
from typing import Optional, Callable

class VoiceController:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
        except Exception as e:
            print(f"VoiceController: Microphone init failed: {e}")
            self.microphone = None
        self.listening = False
        self.command_queue = queue.Queue()
        self.listen_thread = None
        
        # Calibrate for ambient noise
        print("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Microphone calibrated.")
        
        # Voice commands mapping
        self.commands = {
            'draw': 'DRAW',
            'start drawing': 'DRAW',
            'begin': 'DRAW',
            'erase': 'ERASE',
            'clear': 'CLEAR',
            'delete': 'ERASE',
            'undo': 'UNDO',
            'red': 'COLOR_RED',
            'blue': 'COLOR_BLUE',
            'green': 'COLOR_GREEN',
            'black': 'COLOR_BLACK',
            'white': 'COLOR_WHITE',
            'yellow': 'COLOR_YELLOW',
            'save': 'SAVE',
            'export': 'SAVE',
            'stop': 'STOP',
            'quit': 'QUIT',
            'exit': 'QUIT',
            'bigger': 'BRUSH_BIGGER',
            'smaller': 'BRUSH_SMALLER',
            'increase size': 'BRUSH_BIGGER',
            'decrease size': 'BRUSH_SMALLER'
        }
    
    def start_listening(self):
        """Start listening for voice commands in a separate thread."""
        if self.microphone is None:
            print("VoiceController: No microphone available, cannot start listening.")
            return

        if self.listening:
            print("VoiceController: already listening")
            return

        self.listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        print("Voice recognition started. Say commands like 'draw', 'red', 'clear', etc.")
    
    def stop_listening(self):
        """Stop listening for voice commands."""
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1)
        print("Voice recognition stopped.")
    
    def _listen_loop(self):
        """Continuous listening loop running in separate thread."""
        while self.listening:
            try:
                # Listen for audio with timeout
                if self.microphone is None:
                    time.sleep(1)
                    continue

                try:
                    with self.microphone as source:
                        # Listen for phrase with shorter timeout for responsiveness
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                except Exception as e:
                    # Handle audio context errors or concurrent access
                    print(f"VoiceController listen error: {e}")
                    time.sleep(0.2)
                    continue
                
                # Try to recognize speech
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Voice input: '{text}'")
                    
                    # Check for commands
                    command = self._parse_command(text)
                    if command:
                        self.command_queue.put(command)
                        print(f"Voice command recognized: {command}")
                    
                except sr.UnknownValueError:
                    # Could not understand audio - this is normal, continue listening
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results from speech service: {e}")
                    time.sleep(1)  # Wait before retrying
                    
            except sr.WaitTimeoutError:
                # Timeout waiting for audio - this is normal, continue listening
                pass
            except Exception as e:
                print(f"Voice recognition error: {e}")
                time.sleep(1)  # Wait before retrying
    
    def _parse_command(self, text: str) -> Optional[str]:
        """Parse recognized text to extract voice commands."""
        text = text.strip().lower()
        
        # Direct command match
        if text in self.commands:
            return self.commands[text]
        
        # Partial match for phrases
        for phrase, command in self.commands.items():
            if phrase in text:
                return command
        
        return None
    
    def get_command(self) -> Optional[str]:
        """Get the next voice command from the queue."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def has_commands(self) -> bool:
        """Check if there are pending voice commands."""
        return not self.command_queue.empty()
    
    def listen(self) -> Optional[str]:
        """Legacy method for compatibility - returns next command if available."""
        return self.get_command()
