"use client"; // Bắt buộc phải có dòng này ở đầu file

import { useState, KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";

// Định nghĩa kiểu dữ liệu cho tin nhắn
interface Message {
    role: "user" | "bot";
    text: string;
}

const suggestions = ["Cách khắc chế Butterfly?", "Lối lên đồ cho Nakroth?", "Tướng nào đi Mid mạnh nhất?"];

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMsg: Message = { role: "user", text: input };
        const historyPayload = messages.slice(-6).map((msg) => ({
            role: msg.role === "bot" ? "assistant" : "user",
            content: msg.text,
        }));
        setMessages((prev) => [...prev, userMsg]);
        setLoading(true);
        const currentInput = input;
        setInput("");

        try {
            const response = await fetch("http://localhost:8000/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: currentInput,
                    history: historyPayload,
                }),
            });

            if (!response.ok) {
                throw new Error("Backend không phản hồi");
            }

            const data = await response.json();
            const botMsg: Message = { role: "bot", text: data.reply };
            setMessages((prev) => [...prev, botMsg]);
        } catch (error) {
            console.error("Lỗi kết nối Backend:", error);
            setMessages((prev) => [
                ...prev,
                { role: "bot", text: "Lỗi kết nối server rồi ông giáo ơi! Nhớ bật Backend ở port 8000 nhé." },
            ]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    };

    return (
        <div className="flex flex-col h-screen bg-[#0f172a] text-slate-200">
            {/* Header */}
            <header className="py-6 border-b border-slate-800 bg-[#1e293b] shadow-lg text-center">
                <h1 className="text-3xl font-extrabold text-amber-500 tracking-tighter">LIÊN QUÂN AI ASSISTANT ⚔️</h1>
                <p className="text-xs text-slate-400 mt-1 uppercase tracking-widest">Powered by GitHub Models & RAG</p>
            </header>

            {/* Danh sách tin nhắn */}
            <main className="flex-1 overflow-y-auto p-4 md:px-20 space-y-6">
                {messages.length === 0 && (
                    <div className="text-center mt-20 text-slate-500">
                        <p>Hãy hỏi tôi về kỹ năng, vai trò hoặc cách khắc chế các tướng!</p>
                    </div>
                )}

                {messages.map((msg, index) => (
                    <div key={index} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                        <div
                            className={`max-w-[85%] md:max-w-[70%] p-4 rounded-2xl shadow-sm ${
                                msg.role === "user"
                                    ? "bg-amber-600 text-white rounded-tr-none"
                                    : "bg-slate-800 border border-slate-700 rounded-tl-none"
                            }`}
                        >
                            <div className="text-sm md:text-base leading-relaxed">
                                <ReactMarkdown>{msg.text}</ReactMarkdown>
                                </div>
                        </div>
                    </div>
                ))}

                {loading && (
                    <div className="flex justify-start">
                        <div className="bg-slate-800 p-4 rounded-2xl rounded-tl-none animate-pulse text-slate-400 text-sm">
                            Đang phân tích dữ liệu tướng...
                        </div>
                    </div>
                )}
            </main>

            {/* Ô nhập liệu */}
            <footer className="p-4 bg-[#1e293b] border-t border-slate-800">
                <div className="max-w-4xl mx-auto flex gap-2 mb-4 overflow-x-auto ">
                    {suggestions.map((s) => (
                        <button
                            key={s}
                            onClick={() => setInput(s)}
                            className="text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3 py-1 rounded-full whitespace-nowrap"
                        >
                            {s}
                        </button>
                    ))}
                </div>
                <div className="max-w-4xl mx-auto flex gap-3">
                    <input
                        type="text"
                        className="flex-1 p-4 rounded-xl bg-slate-900 border border-slate-700 focus:outline-none focus:ring-2 focus:ring-amber-500 transition-all text-white"
                        placeholder="Nhập câu hỏi... (VD: Butterfly mạnh ở điểm nào?)"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />
                    <button
                        onClick={sendMessage}
                        disabled={loading}
                        className={`px-8 py-4 rounded-xl font-bold uppercase tracking-wider transition-all ${
                            loading
                                ? "bg-slate-700 cursor-not-allowed text-slate-500"
                                : "bg-amber-600 hover:bg-amber-500 text-white shadow-lg shadow-amber-900/20"
                        }`}
                    >
                        {loading ? "..." : "Gửi"}
                    </button>
                </div>
                
            </footer>
        </div>
    );
}
