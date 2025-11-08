import { useState } from 'react';
import './FallingBananas.css';

interface Banana {
  id: number;
  left: number;
  duration: number;
  delay: number;
  size: number;
}

export default function FallingBananas() {
  const [bananas] = useState<Banana[]>(() => {
    // Create 18 bananas with random positions and timings
    const bananaCount = 18;
    const bananas: Banana[] = [];

    for (let i = 0; i < bananaCount; i++) {
      bananas.push({
        id: i,
        left: Math.random() * 100, // 0-100% across screen
        duration: 15 + Math.random() * 10, // 15-25 seconds
        delay: Math.random() * 5, // 0-5 second delay
        size: 20 + Math.random() * 15, // 20-35px
      });
    }

    return bananas;
  });

  return (
    <div className="falling-bananas-container">
      {bananas.map((banana) => (
        <div
          key={banana.id}
          className="falling-banana"
          style={{
            left: `${banana.left}%`,
            animationDuration: `${banana.duration}s`,
            animationDelay: `${banana.delay}s`,
            fontSize: `${banana.size}px`,
          }}
        >
          🍌
        </div>
      ))}
    </div>
  );
}

