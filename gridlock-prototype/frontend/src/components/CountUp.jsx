import { useEffect, useRef, useState } from "react";
import { animate } from "framer-motion";

export default function CountUp({ value = 0 }) {
  const [n, setN] = useState(0);
  const prev = useRef(0);

  useEffect(() => {
    const controls = animate(prev.current, value, {
      duration: 0.8,
      ease: "easeOut",
      onUpdate: (v) => setN(Math.round(v)),
    });
    prev.current = value;
    return () => controls.stop();
  }, [value]);

  return <span>{n}</span>;
}