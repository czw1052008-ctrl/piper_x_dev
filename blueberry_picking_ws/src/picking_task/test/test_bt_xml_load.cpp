#include "behaviortree_cpp/bt_factory.h"
#include "gtest/gtest.h"
#include <filesystem>

TEST(BtXml, LoadsSuctionTree)
{
  BT::BehaviorTreeFactory factory;
  const auto xml = std::filesystem::path(PICKING_BT_XML_SUCTION);
  ASSERT_TRUE(std::filesystem::exists(xml));
  EXPECT_NO_THROW(factory.createTreeFromFile(xml.string()));
}
