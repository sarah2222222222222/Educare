const ApiError = require("../utils/ApiError");
const ApiFeatures = require("../utils/apiFeatures");
const asyncHandler = require("express-async-handler");
const cartModel = require("../models/cartModel");
const courseModel = require("../models/courseModel");
// Create One
exports.createOne = (Model) =>
  asyncHandler(async (req, res, next) => {
    // Restrict post creation to teachers only
    if (Model.modelName === "Post" && req.user.role !== "teacher") {
      return next(new ApiError("Only teachers can create posts", 403));
    }

    // Automatically assign user from token if not already provided
    if (!req.body.user) {
      req.body.user = req.user._id;
    }

    // Ensure user in body matches the authenticated user
    if (req.body.user.toString() !== req.user._id.toString()) {
      return next(
        new ApiError("You are not authorized to perform this action", 403)
      );
    }

    // Inject `class` from URL param if model is Post
    if (Model.modelName === "Post" && req.params.classId) {
      req.body.class = req.params.classId;
    }

    // ---------- Cart Logic ----------
    if (Model.modelName === "Cart") {
      const courseIds = req.body.courseId; // Array of course IDs
      if (!courseIds || !Array.isArray(courseIds) || courseIds.length === 0) {
        return next(new ApiError("At least one course ID is required", 400));
      }

      // Find the courses
      const courses = await courseModel.find({ _id: { $in: courseIds } });
      if (courses.length !== courseIds.length) {
        return next(new ApiError("Some courses not found", 404));
      }

      // Try to find the user's cart
      let cart = await cartModel.findOne({ user: req.user._id });

      if (!cart) {
        // No cart found, create a new one with the courses
        cart = await cartModel.create({
          user: req.user._id,
          course: courseIds, // Add all the course IDs to the cart
          totalAmount: courses.reduce(
            (sum, course) => sum + (course.price || 0),
            0
          ), // Calculate total amount
        });

        // Return the cart data with course prices
        const populatedCart = await cartModel
          .findById(cart._id)
          .populate("course", "price name title");  // Populate course details
        return res.status(201).json({ data: populatedCart });
      }

      // Add new courses to the cart (check if they already exist)
      for (const courseId of courseIds) {
        const exists = cart.course.some(
          (c) => c.toString() === courseId.toString()
        );
        if (!exists) {
          cart.course.push(courseId);
        }
      }

      // Recalculate totalAmount
      const allCourses = await courseModel.find({ _id: { $in: cart.course } });
      cart.totalAmount = allCourses.reduce(
        (sum, course) => sum + (course.price || 0),
        0
      );

      // Save the updated cart
      await cart.save();

      // Return the cart data with course prices
      const populatedCart = await cartModel
        .findById(cart._id)
        .populate("course", "price name title");  // Populate course details
      return res.status(200).json({ data: populatedCart });
    }

    // ---------- Default logic for other models ----------
    const newDoc = await Model.create(req.body);
    res.status(201).json({ data: newDoc });
  });

// Get One (excluding soft-deleted)
exports.getOne = (Model, populateUser = false) =>
  asyncHandler(async (req, res, next) => {
    const { id } = req.params;

    // Initialize the query
    let query = Model.findOne({ _id: id, isDeleted: false });

    // Conditionally populate 'user' field if needed
    if (populateUser) {
      query = query.populate("user", "firstName lastName email");
    }

    // Execute the query to find the document
    const document = await query;

    if (!document) {
      return next(new ApiError(`No document found with ID ${id}`, 404));
    }

    // Convert to plain object to modify safely
    const data = document.toObject();

    // If the model has a `comments` field, filter soft-deleted ones
    if (Array.isArray(data.comments)) {
      data.comments = data.comments.filter((comment) => !comment.deletedAt);
    }

    res.status(200).json({ data });
  });

exports.getByQuery = (Model) =>
  asyncHandler(async (req, res, next) => {
    const query = { ...req.query, isDeleted: false };

    const documents = await Model.find(query);

    if (!documents.length) {
      return next(new ApiError("No documents found matching query", 404));
    }

    // Clean up nested soft-deleted comments, if present
    const data = documents.map((doc) => {
      const obj = doc.toObject();
      if (Array.isArray(obj.comments)) {
        obj.comments = obj.comments.filter((comment) => !comment.deletedAt);
      }
      return obj;
    });

    res.status(200).json({ data });
  });
// Get Many (excluding soft-deleted)
exports.getMany = (Model, modelName = "", populateUser = false) =>
  asyncHandler(async (req, res) => {
    let filter = { isDeleted: false };
    if (req.filterObj) {
      filter = { ...req.filterObj, isDeleted: false };
    }

    const noPagination = req.query.limit === "-1" || !req.query.limit;
    const documentsCount = await Model.countDocuments(filter);

    // Initialize the query
    let query = Model.find(filter);

    // Conditionally apply .populate() based on the populateUser parameter
    if (populateUser) {
      query = query.populate("user", "firstName lastName email");
    }

    // Apply other API features (like search, filter, sort, etc.)
    const apiFeatures = new ApiFeatures(query, req.query)
      .search(modelName)
      .filter()
      .limitFields()
      .sort();

    if (!noPagination) {
      apiFeatures.paginate(documentsCount);
    }

    const { mongooseQuery, paginationResult } = apiFeatures;
    const documents = await mongooseQuery;

    const response = {
      results: documents.length,
      data: documents,
    };

    if (!noPagination) {
      response.paginationResult = paginationResult;
    }

    res.status(200).json(response);
  });

// Update One
exports.updateOne = (Model, options = {}) =>
  asyncHandler(async (req, res, next) => {
    const { requireAuthCheck = false, fieldFilterFn = null } = options;

    // Optional authorization check
    if (
      requireAuthCheck &&
      req.user?._id?.toString() !== req.params.id.toString()
    ) {
      return next(
        new ApiError("You are not authorized to perform this action", 403)
      );
    }

    // Optional: filter allowed update fields
    const updateData = fieldFilterFn ? fieldFilterFn(req.body) : req.body;

    // Apply soft delete condition if the model supports it
    const filter = { _id: req.params.id };
    if (Model.schema.paths.isDeleted) {
      filter.isDeleted = false;
    }

    const document = await Model.findOneAndUpdate(filter, updateData, {
      new: true,
    });

    if (!document) {
      return next(
        new ApiError(`No document found for ID ${req.params.id}`, 404)
      );
    }

    res.status(200).json({ data: document });
  });

// Soft Delete One
exports.deleteOne = (Model) =>
  asyncHandler(async (req, res, next) => {
    const { id } = req.params;

    const document = await Model.findByIdAndUpdate(
      id,
      { isDeleted: true },
      { new: true }
    );

    if (!document) {
      return next(new ApiError(`No document found for ID ${id}`, 404));
    }

    return res.status(200).json({
      message: `Item with ID ${id} has been soft deleted.`,
      data: document,
    });
  });

// Restore (Undo Delete)
exports.restoreOne = (Model) =>
  asyncHandler(async (req, res, next) => {
    const { id } = req.params;

    const document = await Model.findByIdAndUpdate(
      id,
      { isDeleted: false },
      { new: true }
    );

    if (!document) {
      return next(new ApiError(`No deleted document found with ID ${id}`, 404));
    }

    return res.status(200).json({
      message: `Item with ID ${id} has been restored.`,
      data: document,
    });
  });
