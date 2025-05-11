const asyncHandler = require("express-async-handler");
const factory = require("./handlersFactory");
const { v4: uuidv4 } = require("uuid");

const cartModel = require("../models/cartModel");
const User = require("../models/userModel");
const courseModel = require("../models/courseModel");

// Factory exports for basic CRUD operations
exports.getAllcart = factory.getMany(cartModel);
exports.getcart = factory.getOne(cartModel);
exports.updatecart = factory.updateOne(cartModel);
exports.deleteSubmit = factory.deleteOne(cartModel);
exports.createcart = factory.createOne(cartModel);
// Add course to cart
exports.addToCart = async (userId, courseIds) => {
  try {
    // 1. Find the user
    const user = await User.findById(userId);
    if (!user) {
      throw new Error("User not found");
    }

    // 2. Find existing cart or create a new one
    let existingCart = await cartModel.findOne({ user: userId });
    if (!existingCart) {
      // Create a new cart if none exists
      existingCart = await cartModel.create({
        user: userId,
        course: [], // Empty array to hold courses
        totalAmount: 0, // Initial total amount
      });
    }

    // 3. Find the courses based on the provided courseIds
    const courses = await courseModel.find({ '_id': { $in: courseIds } });
    if (courses.length === 0) {
      throw new Error("Courses not found");
    }

    let newTotalAmount = existingCart.totalAmount;

    // 4. Loop through the courses and add them to the cart if not already present
    courses.forEach((course) => {
      const existingCourseInCart = existingCart.course.find(
        (item) => item.toString() === course._id.toString()
      );

      if (!existingCourseInCart) {
        // If the course is not in the cart, add it
        existingCart.course.push(course._id);
        // Add the price of the course to the total amount
        newTotalAmount += course.price;
      } else {
        // If the course is already in the cart, log an informational message
        console.log(`Course ${course.title} already in cart, no action taken.`);
      }
    });

    // 5. Update the total amount for the cart
    existingCart.totalAmount = newTotalAmount;

    // 6. Save the updated cart
    await existingCart.save();

    return existingCart; // Return the updated cart object
  } catch (error) {
    console.error(error);
    throw new Error("Failed to add courses to cart");
  }
};



exports.getUserCart = async (req, res) => {
  try {
    // Retrieve the user ID from the request (e.g., from a JWT token or session)
    const userId = req.user.id;  // Assuming the user is authenticated and their ID is in req.user

    // Find the user's cart
    const userCart = await cartModel.findOne({ user: userId }).populate('course'); // Populating course data

    if (!userCart) {
      return res.status(404).json({ message: 'Cart not found for this user.' });
    }

    // Return the cart with the courses
    res.status(200).json({
      success: true,
      data: userCart,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to retrieve cart' });
  }
};