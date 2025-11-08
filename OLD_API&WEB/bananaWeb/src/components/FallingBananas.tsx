import { useEffect, useState } from 'react';
import './FallingBananas.css';

interface Banana {
  id: number;
  left: number;
  delay: number;
  duration: number;
  size: number;
  rotation: number;
}

export default function FallingBananas() {
  const [bananas, setBananas] = useState<Banana[]>([]);

  useEffect(() => {
    // Create 20 bananas with random properties
    const newBananas: Banana[] = Array.from({ length: 20 }, (_, i) => ({
      id: i,
      left: Math.random() * 100,
      delay: Math.random() * 5,
      duration: 3 + Math.random() * 4, // 3-7 seconds
      size: 20 + Math.random() * 30, // 20-50px
      rotation: Math.random() * 360,
    }));
    setBananas(newBananas);
  }, []);

  return (
    <div className="falling-bananas-container">
      {bananas.map((banana) => (
        <div
          key={banana.id}
          className="falling-banana"
          style={{
            left: `${banana.left}%`,
            animationDelay: `${banana.delay}s`,
            animationDuration: `${banana.duration}s`,
            width: `${banana.size}px`,
            height: `${banana.size}px`,
            transform: `rotate(${banana.rotation}deg)`,
          }}
        >
          🍌
        </div>
      ))}
    </div>
  );
}

