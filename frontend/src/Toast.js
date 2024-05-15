// Toast.js
import React from "react";

const Toast = ({ message }) => {
  return (
    <div className="toast">
      <p className="toast-message">{message}</p>
    </div>
  );
};

export default Toast;
