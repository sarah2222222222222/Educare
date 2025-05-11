const express = require("express");
const dotenv = require("dotenv");
const morgan = require("morgan");
const cors = require("cors");
const dbconnect = require("./config/database");
const ApiError = require("./utils/ApiError");
const globelError = require("./middlewares/errorMiddlware");
const usersroute = require("./routes/userRoutes");
const authsroute = require("./routes/authRoutes");
const classroute = require("./routes/classRoutes");
const postroute = require("./routes/postRoutes");
const courseroute = require("./routes/courseRoutes");
const eventroute = require("./routes/eventRoutes");
const path = require("path");
const graderoute = require("./routes/gradeRoutes ");
const submitRouter = require("./routes/submitRoute");
const cartRoute = require("./routes/cartRoutes");


require('dotenv').config(); // This loads the environment variables
// Load environment variables from .env file
dotenv.config({ path: ".env" });
console.log('PORT:', process.env.PORT); // Log the port to check if it's loaded



// Create the Express app instance
const app = express();

// Use CORS middleware
app.use(cors());
//connecct  to the databbase
dbconnect();

// Middleware - parsing JSON requests
app.use(express.json());

// Use Morgan for logging in development mode
if (process.env.NODE_ENV === "development") {
  app.use(morgan("dev"));
  console.log(`Mode: ${process.env.NODE_ENV}`);
}

// Mount routes with dynamic API version
app.use(`/user`, usersroute);
app.use(`/auth`, authsroute);
app.use(`/class`, classroute);
app.use(`/post`, postroute);
app.use(`/course`, courseroute);
app.use("/uploads", express.static(path.join(__dirname, "uploads")));
app.use(`/event`, eventroute);
app.use(`/grade`, graderoute);
app.use(`/submit`, submitRouter);
app.use(`/cart`, cartRoute);

// Root endpoint for testing
app.all("/", (req, res) => {
  res.send("Welcome to the API!");
});

// Global error handling middleware
app.use(globelError);

// start the server
const PORT = process.env.PORT || 3000;
const server = app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});




// Handle unhandled promise rejections
process.on("unhandledRejection", (err) => {
  console.error(`Unhandled Rejection Error: ${err.name} | ${err.message}`);
  server.close(() => {
    console.error("Shutting down server...");
    process.exit(1);
  });
});

// Handle uncaught exceptions (unhandled errors)
process.on("uncaughtException", (err) => {
  console.error(`Uncaught Exception Error: ${err.name} | ${err.message}`);
  process.exit(1);
});





