const express = require("express");
const {
  getUserValidator,
  createUserValidator,
  updateUserValidator,
  deleteUserValidator,
  changePasswordValidator,
} = require("../utils/validators/UserVaildator");

// Assuming you have a function named 'getCategories' in your services
const {
 
 
  updateUser,
 

  changeUserPassword,

} = require("../services/userService");
const router = express.Router();
const authroute = require("../services/authService");
// Routes


router
  .route("/:id")
  
  .put( authroute.protect, authroute.allowedTo("teacher", "institution","student"), updateUser)


module.exports = router;
