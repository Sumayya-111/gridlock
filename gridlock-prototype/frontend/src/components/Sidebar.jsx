import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import { LayoutDashboard, ScanSearch, Gavel, Settings } from "lucide-react";
import "./Sidebar.css";

const links = [
  { to: "/", label: "Dashboard", Icon: LayoutDashboard },
  { to: "/detect", label: "Detection Studio", Icon: ScanSearch },
  { to: "/judge", label: "Judge Feed", Icon: Gavel },
  { to: "/settings", label: "Settings", Icon: Settings },
];

export default function Sidebar({ open }) {
  return (
    <motion.aside
      className="sidebar"
      initial={false}
      animate={{ width: open ? 220 : 0 }}
      transition={{ type: "spring", stiffness: 320, damping: 34 }}
    >
      <nav className="sidebar-inner">
        {links.map(({ to, label, Icon }) => (
          <NavLink key={to} to={to} end={to === "/"} className="side-link">
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.span
                    layoutId="nav-pill"
                    className="nav-pill"
                    transition={{ type: "spring", stiffness: 400, damping: 32 }}
                  />
                )}
                <Icon size={18} className="nav-icon" />
                <span className="nav-label">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </motion.aside>
  );
}